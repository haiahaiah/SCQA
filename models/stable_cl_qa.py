import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.bert import BertModel, BertPreTrainedModel
from transformers.models.roberta import RobertaModel, RobertaPreTrainedModel
from transformers.models.albert import AlbertModel, AlbertPreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from .cl_qa import Similarity
import sys

sys.path.append("..")
import my_config


def stable_cl_forward(
        model,
        encoder,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
):
    return_dict = return_dict if return_dict is not None else model.config.use_return_dict

    # YH
    # train: input_ids: (batch_size, num_example, seq_length)
    # eval: no num_example dimension
    num_example = 1     # eval mode
    if len(input_ids.shape) == 3:
        # train mode
        num_example = input_ids.size(1)
    batch_size = input_ids.size(0)
    # used for training with augmentation, but code runs ok when evaluate dev or test set
    # should unsqueeze when evaluating in prepare_features

    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1)))  # (batch_size * num_example, seq_length)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1)))
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))

    # if type(model) in [StableCLDistilBertForSpanQA]:
    #     outputs = encoder(
    #         input_ids,
    #         attention_mask=attention_mask,
    #         head_mask=head_mask,
    #         inputs_embeds=inputs_embeds,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )
    #     sequence_output = outputs[0]
    #     pooler_output = sequence_output[:, 0]
    # else:
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
    sequence_output, pooler_output = outputs[0], outputs[1]

    logits = model.qa_outputs(sequence_output)
    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1).contiguous()
    end_logits = end_logits.squeeze(-1).contiguous()

    total_loss = None
    if start_positions is not None and end_positions is not None:
        # YH
        start_positions = start_positions.view(-1)
        end_positions = end_positions.view(-1)

        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2

        aux_stable_loss_weight = my_config.aux_stable_loss_weight
        aux_cl_loss_weight = my_config.aux_cl_loss_weight

        # train with aux loss
        if num_example > 1 and aux_stable_loss_weight != 0:
            # YH: stability. orig + k aug example
            # stability loss from 22-Fine-tuning more stable neural text classifiers for defending word level adversarial attacks.pdf
            start_logits = start_logits.view((batch_size, num_example, start_logits.size(-1)))  # (bs, num_sent, hidden)
            end_logits = end_logits.view((batch_size, num_example, end_logits.size(-1)))  # (bs, num_sent, hidden)

            start_norms, end_norms = torch.tensor([]).to(input_ids.device), torch.tensor([]).to(input_ids.device)
            for i in range(1, num_example):
                # diff of original distribution and ith augmented distribution
                start_diff = start_logits[:, 0] - start_logits[:, i]  # (bs, hidden)
                end_diff = end_logits[:, 0] - end_logits[:, i]

                start_diff_norm = torch.norm(start_diff, dim=-1)  # bs
                end_diff_norm = torch.norm(end_diff, dim=-1)  # bs

                start_norms = torch.cat((start_norms, start_diff_norm.unsqueeze(0)))
                end_norms = torch.cat((end_norms, end_diff_norm.unsqueeze(0)))

            start_norms = start_norms.t()  # (num_example - 1, bs) -> (bs, num_example - 1)
            end_norms = end_norms.t()

            start_stable_loss = torch.norm(start_norms, p=float('inf'), dim=-1)
            end_stable_loss = torch.norm(end_norms, p=float('inf'), dim=-1)

            stability_loss = (torch.mean(start_stable_loss) + torch.mean(end_stable_loss)) / 2
            total_loss += aux_stable_loss_weight * torch.mean(stability_loss)

        if num_example > 1 and aux_cl_loss_weight != 0:
            # YH: contrastive learning loss. orig + 1 aug example
            # compute bs * bs similarity matrix
            # and labels is range(bs), loss_func=CE
            cl_loss = 0.0
            pooler_output = pooler_output.view((batch_size, num_example, pooler_output.size(-1)))  # (bs, num_sent, hidden)
            pooled_1, pooled_2 = pooler_output[:, 0], pooler_output[:, 1]
            cos_sim = model.sim(pooled_1.unsqueeze(1), pooled_2.unsqueeze(0), my_config.temp)
            labels = torch.arange(cos_sim.size(0)).long().to(input_ids.device)

            loss_fct = nn.CrossEntropyLoss()
            cl_loss = loss_fct(cos_sim, labels)
            total_loss += aux_cl_loss_weight * cl_loss

    if not return_dict:
        # output = (start_logits, end_logits) + outputs[2:]
        output = (start_logits, end_logits)
        return ((total_loss,) + output) if total_loss is not None else output

    return QuestionAnsweringModelOutput(
        loss=total_loss,
        start_logits=start_logits,
        end_logits=end_logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


class StableCLBertForSpanQA(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.config = self.config
        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

        self.sim = Similarity()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
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

        return stable_cl_forward(model=self, encoder=self.bert, input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask,
                                 inputs_embeds=inputs_embeds, start_positions=start_positions,
                                 end_positions=end_positions, output_attentions=output_attentions,
                                 return_dict=return_dict,
                                 )


class StableCLRobertaForSpanQA(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        # self.roberta = RobertaModel(config)

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

        self.sim = Similarity()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
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
        return stable_cl_forward(model=self, encoder=self.roberta, input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask,
                                 inputs_embeds=inputs_embeds, start_positions=start_positions,
                                 end_positions=end_positions, output_attentions=output_attentions,
                                 return_dict=return_dict,
                                 )


class StableCLAlbertForSpanQA(AlbertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

        self.sim = Similarity()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
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
        return stable_cl_forward(model=self, encoder=self.albert, input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask,
                                 inputs_embeds=inputs_embeds, start_positions=start_positions,
                                 end_positions=end_positions, output_attentions=output_attentions,
                                 return_dict=return_dict,
                                 )
