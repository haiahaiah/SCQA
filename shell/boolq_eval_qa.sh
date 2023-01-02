cd /home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/

data=boolq
validation_file=/home/yuhai/workspace/qa/data/boolq/dev.jsonl
load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/boolq/

## BERT
#cache_dir=/data/yuhai/cache/huggingface_cache/boolq/
#model_class=stable_cl_bert
#
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/boolq/bert-base-uncased/stable_cl_bert-epochs_10-lr_3e-5-per_bs_12-aug_1-gra_acc_1-stalw_0.0-cllw_0.005-2022-05-12_13:56:37/checkpoint-3930
#model_suffix=bert-base-uncased/stable_cl_bert-epochs_10-lr_3e-5-per_bs_12-aug_1-gra_acc_1-stalw_0.0-cllw_0.005-2022-05-12_13:56:37/checkpoint-3930

cache_dir=/data/yuhai/cache/huggingface_cache/boolq/roberta-base
model_class=stable_cl_roberta

## ROBERTA

model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/boolq/roberta-base/stable_cl_roberta-epochs_5-lr_3e-5-per_bs_12-aug_1-gra_acc_1-stalw_0.005-cllw_0.005-2022-05-12_13:25:22/checkpoint-3930
model_suffix=roberta-base/stable_cl_roberta-epochs_5-lr_3e-5-per_bs_12-aug_1-gra_acc_1-stalw_0.005-cllw_0.005-2022-05-12_13:25:22/checkpoint-3930

model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/boolq/roberta-base/stable_cl_roberta-epochs_8-lr_3e-5-per_bs_12-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-12_16:56:42/checkpoint-3930
model_suffix=roberta-base/stable_cl_roberta-epochs_8-lr_3e-5-per_bs_12-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-12_16:56:42/checkpoint-3930

model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/boolq/roberta-base/stable_cl_roberta-epochs_10-lr_3e-5-per_bs_12-aug_1-gra_acc_1-stalw_0.0001-cllw_0.005-2022-05-12_16:57:22/checkpoint-6288
model_suffix=roberta-base/stable_cl_roberta-epochs_10-lr_3e-5-per_bs_12-aug_1-gra_acc_1-stalw_0.0001-cllw_0.005-2022-05-12_16:57:22/checkpoint-6288


export CUDA_VISIBLE_DEVICES=4

output_dir=./results/"$data"/"$model_suffix"

if [ ! -d "$output_dir"  ];then
  mkdir -p "$output_dir"
else
  echo "$output_dir" exist
fi

export python=/home/yuhai/workspace/anaconda3/envs/hug4.16/bin/python
"$python" ./run_yn_qa.py \
  --model_name_or_path "$model_name_or_path"  \
  --model_class "$model_class" \
  --validation_file "$validation_file" \
  --load_dataset_path "$load_dataset_path" \
  --cache_dir "$cache_dir" \
  --do_eval \
  --per_device_eval_batch_size 128 \
  --output_dir "$output_dir" \
  >> "$output_dir""/log.txt" 2>&1

