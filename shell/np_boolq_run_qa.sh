cd /home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/

dataset_name=np-boolq
train_file=/home/yuhai/workspace/qa/data/np-boolq/train.jsonl
validation_file=/home/yuhai/workspace/qa/data/np-boolq/dev.jsonl
load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/np-boolq/

epochs=20
patience=100

## BERT
#model_name_or_path=bert-base-uncased
#cache_dir=/data/yuhai/cache/huggingface_cache/np-boolq/
#model_class=stable_cl_bert
#lr=3e-5

#model_name_or_path=bert-large-uncased
#cache_dir=/data/yuhai/cache/huggingface_cache/boolq/bert-large
#model_class=stable_cl_bert
#lr=1e-5

## ROBERTA
#model_name_or_path=roberta-base
#cache_dir=/data/yuhai/cache/huggingface_cache/np-boolq/roberta-base
#model_class=stable_cl_roberta
#lr=3e-6

## Albert
model_name_or_path=albert-base-v2
cache_dir=/data/yuhai/cache/huggingface_cache/np-boolq/albert-base-v2
model_class=stable_cl_albert
lr=1e-5

#train bs=12 12269MiB
# train bs=24 18035MiB

#resume_from_checkpoint=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/np-boolq/roberta-base/stable_cl_roberta-epochs_20-patience_30-lr_3e-6-per_bs_6-aug_1-gra_acc_2-stalw_0.0001-cllw_0.005-2022-05-13_14:53:48


## baseline
#aux_stable_loss_weight=0.0
#aux_cl_loss_weight=0.0

# stable
#aux_stable_loss_weight=0.005
#aux_cl_loss_weight=0.0

# cl
#aux_stable_loss_weight=0.0
#aux_cl_loss_weight=0.005

#stable-cl
aux_stable_loss_weight=0.0001
aux_cl_loss_weight=0.005

# stable-cl
#aux_stable_loss_weight=0.001
#aux_cl_loss_weight=0.001

# stable-cl
#aux_stable_loss_weight=0.005
#aux_cl_loss_weight=0.005

#export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=0

#per_bs=12
#aug_data_type=0
#gradient_accumulation_steps=1

per_bs=12
aug_data_type=1
gradient_accumulation_steps=1

#per_bs=4
#aug_data_type=1
#gradient_accumulation_steps=3

# bert-base
# train: per_bs=4, 1, 3 -> 7647MiB
# train: per_bs=12, aug_data_type=0 acc=1  -> 9791MiB
# train: per_bs=12, aug_data_type=1 acc=1  -> 15827MiB;
# bs=128 -> 4909MiB

# bert-large
# train: per_bs=12, aug=0 acc=2 -> 23393MiB

# roberta-base
# train: per_bs=12, aug_data_type=0 acc=1  -> 10025MiB
# train: per_bs=12, aug_data_type=1 acc=1  -> 17793MiB;
# bs=128 -> MiB


output_dir=./results/"$dataset_name"/"$model_name_or_path"/"$model_class"-epochs_"$epochs"-patience_"$patience"-lr_"$lr"-per_bs_"$per_bs"-aug_"$aug_data_type"-gra_acc_"$gradient_accumulation_steps"-stalw_"$aux_stable_loss_weight"-cllw_"$aux_cl_loss_weight"-$(date "+%Y-%m-%d_%H:%M:%S")

if [ ! -d "$output_dir"  ];then
  mkdir -p "$output_dir"
else
  echo "$output_dir" exist
fi

export python=/home/yuhai/workspace/anaconda3/envs/hug4.16/bin/python
"$python" ./run_yn_qa.py \
  --model_name_or_path "$model_name_or_path"  \
  --model_class "$model_class" \
  --aux_stable_loss_weight "$aux_stable_loss_weight" \
  --aux_cl_loss_weight "$aux_cl_loss_weight" \
  --train_file "$train_file" \
  --validation_file "$validation_file" \
  --load_dataset_path "$load_dataset_path" \
  --cache_dir "$cache_dir" \
  --aug_data_type "$aug_data_type" \
  --do_train \
  --do_eval \
  --per_device_train_batch_size "$per_bs" \
  --gradient_accumulation_steps "$gradient_accumulation_steps" \
  --per_device_eval_batch_size 128 \
  --learning_rate "$lr" \
  --num_train_epochs "$epochs" \
  --output_dir "$output_dir" \
  --evaluation_strategy 'steps' \
  --eval_steps 500 \
  --save_total_limit 3 \
  --patience "$patience" \
  --metric_for_best_model 'accuracy' \
  --load_best_model_at_end \
  >> "$output_dir""/log.txt" 2>&1

