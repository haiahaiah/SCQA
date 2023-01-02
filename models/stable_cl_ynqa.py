import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.bert import BertModel, BertPreTrainedModel
from transformers.models.roberta import RobertaModel, RobertaPreTrainedModel
from transformers.models.albert import AlbertModel, AlbertPreTrainedModel

from transformers.modeling_outputs import MultipleChoiceModelOutput

from .cl_qa import Similarity
import sys

sys.path.append("..")
import my_config


def stable_cl_forward_for_yes_or_no(
        model,
        encoder,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
):
    r"""
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
                num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
                `input_ids` above)
            """
    return_dict = return_dict if return_dict is not None else model.config.use_return_dict

    # YH
    # size of input_ids should be (batch_size, num_example, num_choices, seq_length)
    batch_size = input_ids.size(0)
    num_example = input_ids.size(1)

    input_ids = input_ids.view((-1, input_ids.size(-1))) if input_ids is not None else None
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) if attention_mask is not None else None
    token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) if token_type_ids is not None else None
    position_ids = position_ids.view((-1, position_ids.size(-1))) if position_ids is not None else None
    inputs_embeds = (
        inputs_embeds.view((-1, inputs_embeds.size(-2), inputs_embeds.size(-1)))
        if inputs_embeds is not None
        else None
    )

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    pooled_output = outputs[1]

    pooled_output = model.dropout(pooled_output)
    logits = model.softmax(model.classifier(pooled_output))

    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss()
        labels = labels.view(-1)
        # if len(labels.shape) == 2:
        #     labels = labels.squeeze(1)
        # print('reshaped_logits.size(): ', reshaped_logits.size(), 'labels.size(): ', labels.size())
        loss = loss_fct(logits, labels)

        aux_stable_loss_weight = my_config.aux_stable_loss_weight
        aux_cl_loss_weight = my_config.aux_cl_loss_weight

        if num_example > 1 and aux_stable_loss_weight != 0:
            # YH: stability loss. orig + k aug example
            reshaped_logits = logits.view((batch_size, num_example, logits.size(-1)))
            norms = torch.tensor([]).to(input_ids.device)
            for i in range(1, num_example):
                # diff of original distribution and ith augmented distribution
                diff = reshaped_logits[:, 0] - reshaped_logits[:, i]  # (bs, hidden)
                diff_norm = torch.norm(diff, dim=-1)  # bs
                norms = torch.cat((norms, diff_norm.unsqueeze(0)))

            norms = norms.t()  # (num_example - 1, bs) -> (bs, num_example - 1)
            stability_loss = torch.norm(norms, p=float('inf'), dim=-1)
            loss += aux_stable_loss_weight * torch.mean(stability_loss)
        # YH: contrastive learning loss. orig + 1 aug example
        # compute bs * bs similarity matrix
        # and labels is range(bs), loss_func=CE
        if num_example > 1 and aux_cl_loss_weight != 0:
            pooled_output = pooled_output.view(
                (batch_size, num_example, pooled_output.size(-1)))  # (bs, num_sent, hidden)
            pooled_1, pooled_2 = pooled_output[:, 0], pooled_output[:, 1]
            cos_sim = model.sim(pooled_1.unsqueeze(1), pooled_2.unsqueeze(0), my_config.temp)
            cl_labels = torch.arange(cos_sim.size(0)).long().to(input_ids.device)

            loss_fct = nn.CrossEntropyLoss()
            cl_loss = loss_fct(cos_sim, cl_labels)
            loss += aux_cl_loss_weight * cl_loss

    # print('loss: ', loss.cpu().data)
    if not return_dict:
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return MultipleChoiceModelOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


class StableCLBertForYesNo(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)
        self.sim = Similarity()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return stable_cl_forward_for_yes_or_no(self, self.bert, input_ids=input_ids,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask,
                                               inputs_embeds=inputs_embeds,
                                               labels=labels,
                                               output_attentions=output_attentions,
                                               output_hidden_states=output_hidden_states,
                                               return_dict=return_dict, )


class StableCLRobertaForYesNo(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)
        self.sim = Similarity()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            labels=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return stable_cl_forward_for_yes_or_no(self, self.roberta, input_ids=input_ids,
                                                     attention_mask=attention_mask,
                                                     token_type_ids=token_type_ids,
                                                     position_ids=position_ids,
                                                     head_mask=head_mask,
                                                     inputs_embeds=inputs_embeds,
                                                     labels=labels,
                                                     output_attentions=output_attentions,
                                                     output_hidden_states=output_hidden_states,
                                                     return_dict=return_dict, )


class StableCLAlbertForYesNo(AlbertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)
        self.sim = Similarity()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return stable_cl_forward_for_yes_or_no(self, self.albert, input_ids=input_ids,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask,
                                               inputs_embeds=inputs_embeds,
                                               labels=labels,
                                               output_attentions=output_attentions,
                                               output_hidden_states=output_hidden_states,
                                               return_dict=return_dict, )
