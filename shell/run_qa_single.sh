
dataset_name=squad
train_file=/home/yuhai/workspace/qa/data/SQuAD/train-v1.1.json
validation_file=/home/yuhai/workspace/qa/data/SQuAD/dev-v1.1.json
load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/squad/squad.py

model_name_or_path=bert-base-uncased
model_class=baseline_bert  # stable cl
bs=12
aug_data_type=0

output_dir=./results/"$dataset_name"/"$model_name_or_path"/"$model_class"-bs_"$bs"-aug_data_type_"$aug_data_type"-$(date "+%Y-%m-%d_%H:%M:%S")

if [ ! -d "$output_dir"  ];then
  mkdir -p "$output_dir"
else
  echo "$output_dir" exist
fi

export CUDA_VISIBLE_DEVICES=0

export python=/home/yuhai/workspace/anaconda3/envs/hug4.16/bin/python
"$python" ./run_qa.py \
  --model_name_or_path "$model_name_or_path"  \
  --model_class "$model_class" \
`  #  --dataset_name "$dataset_name" \
` \
  --train_file "$train_file" \
  --validation_file "$validation_file" \
  --load_dataset_path "$load_dataset_path" \
  --do_train \
  --aug_data_type "$aug_data_type" \
  --do_eval \
  --per_device_train_batch_size 3 \
  --per_device_eval_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --output_dir "$output_dir" \
  >> "$output_dir""/log.txt" 2>&1
