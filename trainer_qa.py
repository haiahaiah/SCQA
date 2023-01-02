# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A subclass of `Trainer` specific to Question-Answering tasks
"""
import os

import torch
import torch.nn.functional as F
from transformers import Trainer, is_torch_tpu_available, EvalPrediction
from transformers.trainer_utils import PredictionOutput


if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            # span qa or mc qa
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        elif self.post_process_function is None and self.compute_metrics is not None:
            # multiple choice and y/n qa historically
            metrics = self.compute_metrics(EvalPrediction(predictions=output.predictions.argmax(axis=1), label_ids=output.label_ids.squeeze(1)))
            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def sent_emb_eval(self, eval_dataset=None, eval_examples=None, dropout=False, save_vec_path='./pooler_output.pt'):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        def align_loss(x, y, alpha=2):
            return (x - y).norm(p=2, dim=1).pow(alpha).mean()

        def uniform_loss(x, x1, x2, t=2):
            n = x.shape[0]
            pow_c = 2
            all_pair = torch.pdist(x, p=2).pow(pow_c).mul(-t).exp()
            pos_pair = (x1 - x2).norm(p=2, dim=1).pow(pow_c).mul(-t).exp()
            return ((all_pair.sum() - pos_pair.sum()) / (n * (n - 1) / 2 - n / 2)).log()

        align, uniform = 0.0, 0.0
        if os.path.isfile(save_vec_path):
            pooler_output_all = torch.load(save_vec_path)
            z1, z2 = pooler_output_all[:, 0], pooler_output_all[:, 1]
            z1 = F.normalize(z1, p=2, dim=1)
            z2 = F.normalize(z2, p=2, dim=1)
            z = torch.cat((z1, z2))
            uniform = uniform_loss(z, z1, z2, t=2)
            align = align_loss(z1, z2, alpha=2)
        else:
            uniform_all, align_all = [], []
            pooler_output_all = torch.tensor([]).to(self.args.device)
            self.model.eval()
            with torch.no_grad():
                for step, inputs in enumerate(eval_dataloader):
                    for k, v in inputs.items():
                        inputs[k] = v.to(self.args.device)
                    raw_outputs = self.model(**inputs, output_hidden_states=True, return_dict=True, sent_emb=True, dropout=dropout)
                    pooler_output = raw_outputs.pooler_output
                    pooler_output_all = torch.cat((pooler_output_all, pooler_output))
                    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
                    z1 = F.normalize(z1, p=2, dim=1)
                    z2 = F.normalize(z2, p=2, dim=1)
                    z = torch.cat((z1, z2))
                    uniform_all.append(uniform_loss(z, z1, z2, t=2))
                    align_all.append(align_loss(z1, z2, alpha=2))
            torch.save(pooler_output_all, save_vec_path)
            align = sum(align_all) / len(align_all)
            uniform = sum(uniform_all) / len(uniform_all)
        return {'align': float(align.cpu().numpy()), 'uniform': float(uniform.cpu().numpy())}

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(predict_examples, predict_dataset, output.predictions, "predict")
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)
