
dataset_name=natural_questions
train_file=/home/yuhai/workspace/qa/data/squad_v2/train-v2.0.json
validation_file=/home/yuhai/workspace/qa/data/squad_v2/dev-v2.0.json
load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/squad_v2/

## BERT
#model_name_or_path=bert-base-uncased
#model_name_or_path=bert-large-uncased   # bs=8 12123MiB

#model_class=baseline_bert  # baseline_model stable_model cl_model
#model_class=cl_bert
#model_class=stable_bert

## ROBERTA
model_name_or_path=roberta-base     # bs=6 6949 bs=12 10110 bs=24 16065
#model_name_or_path=roberta-large    # bs=8 17371MiB; bs=12 23663MiB

model_class=baseline_roberta
#model_class=stable_roberta
#model_class=cl_roberta

aux_loss_weight=0.0

#export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=5
NUM_GPU=1
per_bs=24
aug_data_type=0
gradient_accumulation_steps=1

# batch_size=12
#output_dir=./results/"$dataset_name"/"$model_name_or_path"/"$model_class"-bs_12-aug_data_type_"$aug_data_type"-$(date "+%Y-%m-%d_%H:%M:%S")

# batch_size=24
output_dir=./results/"$dataset_name"/"$model_name_or_path"/"$dataset_name"-"$model_class"-bs_24-aug_data_type_"$aug_data_type"-$(date "+%Y-%m-%d_%H:%M:%S")

if [ ! -d "$output_dir"  ];then
  mkdir -p "$output_dir"
else
  echo "$output_dir" exist
fi

export python=/home/yuhai/workspace/anaconda3/envs/hug4.16/bin/python
"$python" ./run_qa.py \
  --model_name_or_path "$model_name_or_path"  \
  --model_class "$model_class" \
  --aux_loss_weight "$aux_loss_weight" \
`  #  --dataset_name "$dataset_name" \
` \
  --version_2_with_negative \
  --train_file "$train_file" \
  --validation_file "$validation_file" \
  --load_dataset_path "$load_dataset_path" \
  --do_train \
  --aug_data_type "$aug_data_type" \
  --do_eval \
  --per_device_train_batch_size "$per_bs" \
  --gradient_accumulation_steps "$gradient_accumulation_steps" \
  --per_device_eval_batch_size 128 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --output_dir "$output_dir" \
  >> "$output_dir""/log.txt" 2>&1

