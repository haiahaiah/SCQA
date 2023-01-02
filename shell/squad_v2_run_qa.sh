
dataset_name=squad_v2
train_file=/home/yuhai/workspace/qa/data/squad_v2/train-v2.0.json
validation_file=/home/yuhai/workspace/qa/data/squad_v2/dev-v2.0.json
load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/squad_v2/

lr=3e-5
lr=1e-5
## BERT
model_name_or_path=bert-base-uncased
#model_name_or_path=bert-large-uncased   # bs=8 12123MiB

#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/bert-base-uncased/stable_cl_bert-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-26_17:09:18

#model_class=baseline_bert  # baseline_model stable_model cl_model
#model_class=cl_bert
#model_class=stable_bert
model_class=stable_cl_bert

## ROBERTA
#model_name_or_path=roberta-base     # bs=6 6949 bs=12 10110 bs=24 16065
#model_name_or_path=roberta-large    # bs=8 17371MiB; bs=12 23663MiB

#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/roberta-base/stable_cl_roberta-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-21_11:02:17

#model_class=baseline_roberta
#model_class=stable_roberta
#model_class=cl_roberta
#model_class=stable_cl_roberta
#
### Albert
model_name_or_path=albert-base-v2
cache_dir=/data/yuhai/cache/huggingface_cache/squad-2/albert-base-v2
model_class=stable_cl_albert

# train bs=12 8235MiB
# train bs=24 14157MiB

# baseline
aux_stable_loss_weight=0.0
aux_cl_loss_weight=0.0

# stable-cl
#aux_stable_loss_weight=0.001
#aux_cl_loss_weight=0.001

#export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=0
per_bs=12
aug_data_type=0
gradient_accumulation_steps=1

#per_bs=12
#aug_data_type=1
#gradient_accumulation_steps=1

# batch_size=12 or 24
output_dir=./results/"$dataset_name"/"$model_name_or_path"/"$model_class"-lr-"$lr"-per_bs_"$per_bs"-aug_"$aug_data_type"-gacc_"$gradient_accumulation_steps"-stalw_"$aux_stable_loss_weight"-cllw_"$aux_cl_loss_weight"-$(date "+%Y-%m-%d_%H:%M:%S")

if [ ! -d "$output_dir"  ];then
  mkdir -p "$output_dir"
else
  echo "$output_dir" exist
fi

export python=/home/yuhai/workspace/anaconda3/envs/hug4.16/bin/python
"$python" ./run_qa.py \
  --model_name_or_path "$model_name_or_path"  \
  --model_class "$model_class" \
  --aux_stable_loss_weight "$aux_stable_loss_weight" \
  --aux_cl_loss_weight "$aux_cl_loss_weight" \
  --version_2_with_negative \
  --train_file "$train_file" \
  --validation_file "$validation_file" \
  --load_dataset_path "$load_dataset_path" \
  --cache_dir "$cache_dir" \
  --do_train \
  --aug_data_type "$aug_data_type" \
  --do_eval \
  --per_device_train_batch_size "$per_bs" \
  --gradient_accumulation_steps "$gradient_accumulation_steps" \
  --per_device_eval_batch_size 128 \
  --learning_rate "$lr" \
  --num_train_epochs 3 \
  --output_dir "$output_dir" \
  >> "$output_dir""/log.txt" 2>&1 \
  --fp16 \

