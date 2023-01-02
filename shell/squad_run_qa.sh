cd /home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/

dataset_name=squad
train_file=/home/yuhai/workspace/qa/data/SQuAD/train-v1.1.json
validation_file=/home/yuhai/workspace/qa/data/SQuAD/dev-v1.1.json
load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/squad/

lr=1e-5
epoch=3
seed=42

## BERT
model_name_or_path=bert-base-uncased   # bs=24 15297MiB
cache_dir=/data/yuhai/cache/huggingface_cache/squad/bert-base-uncased

#model_name_or_path=bert-large-uncased

#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad/bert-base-uncased/stable_cl_bert-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-26_17:36:45

#model_class=baseline_bert
#model_class=cl_bert
#model_class=stable_bert
model_class=stable_cl_bert

## ROBERTA
#model_name_or_path=roberta-base      # bs=24 15573MiB
#model_name_or_path=roberta-large    # bs=8 17371MiB;

#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad/roberta-base/stable_cl_roberta-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-26_17:37:08

#model_class=baseline_roberta
#model_class=stable_roberta
#model_class=cl_roberta
#model_class=stable_cl_roberta

### Albert
#model_name_or_path=albert-base-v2
#cache_dir=/data/yuhai/cache/huggingface_cache/squad-2/albert-base-v2
#model_class=stable_cl_albert

#aux_stable_loss_weight=0.0
#aux_cl_loss_weight=0.0

aux_stable_loss_weight=0.000001
aux_cl_loss_weight=0.1

#export CUDA_VISIBLE_DEVICES=3,2,1,0
export CUDA_VISIBLE_DEVICES=2
#NUM_GPU=1
per_bs=12
aug_data_type=1
gradient_accumulation_steps=1

#per_bs=12
#aug_data_type=1
#gradient_accumulation_steps=1

root_dir=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results
output_dir="$root_dir"/"$dataset_name"/"$model_name_or_path"/"$model_class"-seed_"$seed"-epoch_"$epoch"-lr_"$lr"-per_bs_"$per_bs"-aug_data_type_"$aug_data_type"-gc_"$gradient_accumulation_steps"-stalw_"$aux_stable_loss_weight"-cllw_"$aux_cl_loss_weight"-$(date "+%Y-%m-%d_%H:%M:%S")

if [ ! -d "$output_dir"  ];then
  mkdir -p "$output_dir"
else
  echo "$output_dir" exist
fi

export python=/home/yuhai/workspace/anaconda3/envs/hug4.16/bin/python
"$python" ./run_qa.py \
  --model_name_or_path "$model_name_or_path"  \
  --model_class "$model_class" \
  --seed "$seed" \
  --train_file "$train_file" \
  --validation_file "$validation_file" \
  --load_dataset_path "$load_dataset_path" \
  --do_train \
  --cache_dir "$cache_dir" \
  --aug_data_type "$aug_data_type" \
  --do_eval \
  --save_steps 10000 \
  --per_device_train_batch_size "$per_bs" \
  --gradient_accumulation_steps "$gradient_accumulation_steps" \
  --per_device_eval_batch_size 128 \
  --learning_rate "$lr" \
  --num_train_epochs "$epoch" \
  --output_dir "$output_dir" \
  >> "$output_dir""/log.txt" 2>&1
