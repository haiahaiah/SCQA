#!/usr/bin/env python
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
Fine-tuning the library models for question answering using a slightly adapted version of the 🤗 Trainer.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import logging
import os
import sys
import json

from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset, load_metric

import torch
import transformers
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from utils_qa import postprocess_mc_or_yn_qa_predictions
from models.stable_cl_mcqa import StableCLBertForMultipleChoice, StableCLRobertaForMultipleChoice, StableCLAlbertForMultipleChoice

import my_config

MODEL_CLASS = {
    'stable_cl_bert': StableCLBertForMultipleChoice,
    'stable_cl_roberta': StableCLRobertaForMultipleChoice,
    'stable_cl_albert': StableCLAlbertForMultipleChoice,

}

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.16.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = logging.getLogger(__name__)

# os.environ["CUDA_VISIBLE_DEVICES"] = '3'

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_class: str = field(
        default="baseline_bert", metadata={"help": "model class such as baseline, stable and cl"}
    )
    dropout: bool = field(
        default=False, metadata={"help": "dropout"}
    )
    aux_stable_loss_weight: float = field(
        default=0.0, metadata={"help": "aux_stable_loss_weight "}
    )
    aux_cl_loss_weight: float = field(
        default=0.0, metadata={"help": "aux_cl_loss_weight "}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default='/data/yuhai/cache/huggingface_cache',
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    load_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The script of loading dataset containing dataset builder"}
    )
    name: Optional[str] = field(
        default='all', metadata={"help": "version of high middle or all"}
    )
    num_choices: Optional[int] = field(
        default=4,
    )
    patience: Optional[int] = field(
        default=5, metadata={"help": "early stopping patience"}
    )
    data_dir: Optional[str] = field(
        default='', metadata={"help": "used for reclor dataset"}
    )
    train_folder: Optional[str] = field(default=None, metadata={
        "help": "The input training data file (a folder contains some text file)."})
    aug_data_type: Optional[int] = field(default=0, metadata={
        "help": "how to augment training data, 0: no aug, 1: drop twice, 2: aug by id"})
    validation_folder: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    do_sent_emb: Optional[bool] = field(
        default=False,
        metadata={"help": "compute sent embedding for uniform and alignment"}
    )
    adv_file: Optional[str] = field(
        default=None,
        metadata={"help": "adv_file"},
    )
    adv_type: Optional[int] = field(
        default=0,
        metadata={"help": "0: dropout as aug, 1: adv_file"},
    )
    test_folder: Optional[str] = field(
        default=None,
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
                    "be faster on GPU but will be slower on TPU)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )

    # def __post_init__(self):
    #     if (
    #         self.dataset_name is None
    #         and self.train_file is None
    #         and self.validation_file is None
    #         and self.test_file is None
    #     ):
    #         raise ValueError("Need either a dataset name or a training/validation file/test_file.")
    #     else:
    #         if self.train_file is not None:
    #             extension = self.train_file.split(".")[-1]
    #             assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
    #         if self.validation_file is not None:
    #             extension = self.validation_file.split(".")[-1]
    #             assert extension in ["csv", "json"] or self.validation_file in ['squad_adv_addsent', 'squad_adv_addonesent'], "`validation_file` should be a csv or a json file."
    #         if self.test_file is not None:
    #             extension = self.test_file.split(".")[-1]
    #             assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    my_config.aux_stable_loss_weight = model_args.aux_stable_loss_weight
    my_config.aux_cl_loss_weight = model_args.aux_cl_loss_weight
    print('aux_stable_loss_weight: ', my_config.aux_stable_loss_weight, 'aux_stable_loss_weight: ',
          my_config.aux_cl_loss_weight)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    # if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    #     if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
    #         raise ValueError(
    #             f"Output directory ({training_args.output_dir}) already exists and is not empty. "
    #             "Use --overwrite_output_dir to overcome."
    #         )
    #     elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #         logger.info(
    #             f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
    #             "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
    #         )

    # write args to output_dir/args.txt
    with open(os.path.join(training_args.output_dir, "args.txt"), 'w') as args_file:
        args_file.write(f"data_args: {data_args}\n")
        args_file.write(f"model_args: {model_args}\n")
        args_file.write(f"training_args: {training_args}\n")
        args_file.write(f"cuda device count: {torch.cuda.device_count()}\n")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        data_files = {}
        if data_args.train_folder is not None:
            data_files["train"] = data_args.train_folder + "**"
        if data_args.validation_folder is not None:
            data_files["validation"] = data_args.validation_folder + "**"
        if data_args.test_folder is not None:
            data_files["test"] = data_args.test_folder + "**"

        # local file
        if 'reclor' in data_args.data_dir and data_args.train_folder is None:
            print('data_dir:', data_args.data_dir)
            raw_datasets = load_dataset(data_args.load_dataset_path, data_dir=data_args.data_dir,
                                        cache_dir=model_args.cache_dir)
        else:
            print('data_files: ', data_files)
            raw_datasets = load_dataset(data_args.load_dataset_path, data_files=data_files, cache_dir=model_args.cache_dir, ignore_verifications=True)
    global id_for_print
    id_for_print = 0
    if data_args.adv_file is not None:
        adv_dict = {}   # id -> {'c': c, 'q': q, 'a': a, 'options': options}
        # adv_type=0: dropout in prepare_validation_features
        # adv_type=1: read adv data and save it to use in prepare_validation_features
        if "reclor" in data_args.adv_file:
            with open(data_args.adv_file, 'r') as f:
                data = json.load(f)
                for id_, row in enumerate(data):
                    adv_dict[row['id_string']] = {
                        'q': row['question'],
                        'a': str(row["label"]),
                        'c': row["context"],
                        'options': row["answers"]
                    }
        elif "race" in data_args.adv_file:
            with open(data_args.adv_file, 'r') as f:
                adv_type = data_args.adv_type
                assert adv_type in [1]
                if adv_type == 1:
                    # all examples in one file like adv-race
                    data = json.load(f)
                    for id_, value in data.items():
                        if 'middle' in id_:
                            id_ = 'middle-' + id_.split('/')[-1]
                        if 'high' in id_:
                            id_ = 'high-' + id_.split('/')[-1]

                        adv_dict[id_] = {
                            'q': value['question'],
                            'a': str(value['label']),
                            'c': value['context'],
                            'options': value["options"]
                        }

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        # aux_stable_loss_weight=model_args.aux_stable_loss_weight,
        # aux_cl_loss_weight=model_args.aux_cl_loss_weight,
        # return_unused_kwargs=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if training_args.resume_from_checkpoint is not None:
        model = MODEL_CLASS[model_args.model_class].from_pretrained(
            training_args.resume_from_checkpoint,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = MODEL_CLASS[model_args.model_class].from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval or data_args.do_sent_emb:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names
    example_id_name = "id" if "id" in column_names else column_names[0]
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answer" if "answer" in column_names else column_names[2]
    options_column_name = "options" if "options" in column_names else column_names[3]
    num_choices = data_args.num_choices

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def form_instance_from_example(examples):
        # concat question and option in options
        instances = {
            example_id_name: [],
            question_column_name: [],
            context_column_name: [],
            answer_column_name: []
        }
        for i in range(len(examples[example_id_name])):
            question = examples[question_column_name][i]
            context = examples[context_column_name][i]
            answer = examples[answer_column_name][i]
            options = examples[options_column_name][i]  # option list
            assert len(options) == num_choices

            for option in options:
                question_option = question + '. ' + option
                instances[example_id_name].append(examples[example_id_name][i])
                instances[question_column_name].append(question_option)
                instances[context_column_name].append(context)
                instances[answer_column_name].append(answer)
        instances[question_column_name] = [q.lstrip() for q in instances[question_column_name]]
        return instances

    # Training preprocessing
    def prepare_train_features(examples):
        instances = form_instance_from_example(examples)
        tokenized_instances = tokenizer(
            instances[question_column_name if pad_on_right else context_column_name],
            instances[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            # stride=data_args.doc_stride,      # truncate temporarily
            # return_overflowing_tokens=True,
            # return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )
        # one example -> one tokenized instance

        # YH: augmented examples
        # 0. no aug
        # 1. copy examples and concat to achieve drop twice
        # 2. get corresponding augmented examples according id
        aug_data_type = data_args.aug_data_type
        features = {}
        if aug_data_type == 0:
            total_cnt = len(tokenized_instances['input_ids'])
            for key in tokenized_instances:
                features[key] = [[tokenized_instances[key][i * num_choices:(i + 1) * num_choices]] for i in
                                 range(total_cnt // num_choices)]
            features['labels'] = [[int(examples[answer_column_name][i])] for i in
                                  range(total_cnt // num_choices)]
            features[example_id_name] = [[examples[example_id_name][i]] for i in
                                         range(total_cnt // num_choices)]

        elif aug_data_type == 1:
            total_cnt = len(tokenized_instances['input_ids'])
            for key in tokenized_instances:
                features[key] = [[tokenized_instances[key][i * num_choices:(i + 1) * num_choices],
                                  tokenized_instances[key][i * num_choices:(i + 1) * num_choices]] for i in
                                 range(total_cnt // num_choices)]
            features['labels'] = [[int(examples[answer_column_name][i]), int(examples[answer_column_name][i])] for i in
                                  range(total_cnt // num_choices)]
            features[example_id_name] = [[examples[example_id_name][i], examples[example_id_name][i]] for i in
                                         range(total_cnt // num_choices)]
        else:
            features = tokenized_instances
        return features

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if argument is specified
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        # Create train feature from dataset
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                prepare_train_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            # Running tokenizer on train dataset:   0%|                | 0/88 [00:00<?, ?ba/s]03/10/2022 11:00:25 - INFO - datasets.arrow_dataset - Caching processed dataset at /home/yuhai/.cache/huggingface/datasets/squad/plain_text/1.0.0/d6ec3ceb99ca480ce37cdd35555d6cb2511d223b9150cce08a837ef62ffea453/cache-e217b79150752e18.arrow
        if data_args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    # Validation preprocessing
    def prepare_validation_features(examples):
        adv_type = data_args.adv_type
        if adv_type == 0:
            examples[example_id_name] += examples[example_id_name]
            examples[question_column_name] += examples[question_column_name]
            examples[context_column_name] += examples[context_column_name]
            examples[answer_column_name] += examples[answer_column_name]
        elif adv_type == 1:
            qids = [id_ for id_ in examples[example_id_name]]
            for i, qid in enumerate(qids):
                assert qid in adv_dict
                examples[example_id_name] += [qid]
                examples[question_column_name] += [adv_dict[qid]["q"]]
                examples[context_column_name] += [adv_dict[qid]["c"]]
                examples[answer_column_name] += [adv_dict[qid]["a"]]
                examples[options_column_name] += [adv_dict[qid]["options"]]
                global id_for_print
                if id_for_print < 2:
                    print('\nqid: %s,\n\t\taug_context: %s,\n\t\t,orig_context: %s' % (
                    qid, adv_dict[qid]["c"], examples[context_column_name][i]))
                id_for_print += 1
        else:
            pass
        instances = form_instance_from_example(examples)
        tokenized_instances = tokenizer(
            instances[question_column_name if pad_on_right else context_column_name],
            instances[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            # stride=data_args.doc_stride,      # truncate temporarily
            # return_overflowing_tokens=True,
            # return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )
        features = {}
        # TODO check
        # should be ok since the labels is ok
        total_cnt = len(tokenized_instances['input_ids']) // 2
        for key in tokenized_instances:
            features[key] = [[tokenized_instances[key][i * num_choices:(i + 1) * num_choices],
                              tokenized_instances[key][total_cnt + i * num_choices : total_cnt + (i + 1) * num_choices]] for i in
                             range(total_cnt // num_choices)]
        example_cnt = total_cnt // num_choices
        features['labels'] = [[int(examples[answer_column_name][i]), int(examples[answer_column_name][i + example_cnt])] for i in
                              range(example_cnt)]
        features[example_id_name] = [[examples[example_id_name][i], examples[example_id_name][i + example_cnt]] for i in
                                     range(example_cnt)]

        return features

    if training_args.do_eval or data_args.do_sent_emb:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval or --do_sent_emb requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            eval_examples = eval_examples.select(range(data_args.max_eval_samples))
        # Validation Feature Creation
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        # Predict Feature Creation
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    def post_processing_function(examples, features, predictions, stage="eval"):
        eval_prediction = postprocess_mc_or_yn_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            output_dir=training_args.output_dir,
            prefix=stage,
        )
        return eval_prediction

    metric = load_metric("./metrics/accuracy")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Initialize our Trainer
    tb_writer = SummaryWriter(os.path.join(training_args.output_dir, 'writer'))
    if training_args.load_best_model_at_end:
        trainer = QuestionAnsweringTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train or data_args.do_sent_emb else None,
            eval_dataset=eval_dataset if training_args.do_eval or data_args.do_sent_emb else None,
            eval_examples=eval_examples if training_args.do_eval or data_args.do_sent_emb else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=post_processing_function,
            compute_metrics=compute_metrics,
            callbacks=[TensorBoardCallback(tb_writer), EarlyStoppingCallback(early_stopping_patience=data_args.patience)] if training_args.do_train else None
        )
    else:
        trainer = QuestionAnsweringTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            eval_examples=eval_examples if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=post_processing_function,
            compute_metrics=compute_metrics,
            callbacks=[TensorBoardCallback(tb_writer)] if training_args.do_train else None
        )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        print('start training from checkpoint: ', checkpoint)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluate uniform and align
    if data_args.do_sent_emb:
        save_vec_path = os.path.join(training_args.output_dir, 'pooler_output.pt')
        logger.info("*** Evaluate Uniformity and alignment ***")
        metrics = trainer.sent_emb_eval(eval_dataset=eval_dataset, eval_examples=eval_examples, dropout=model_args.dropout, save_vec_path=save_vec_path)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        prefix = "analyze_wo_dropout"
        if model_args.dropout:
            prefix = "analyze_w_dropout"
        trainer.log_metrics(prefix, metrics)
        trainer.save_metrics(prefix, metrics)
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(predict_dataset, predict_examples)
        metrics = results.metrics

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "question-answering"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
