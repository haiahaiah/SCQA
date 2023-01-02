
cd /home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/

dataset_name=boolq_with_mix_same_in_np_boolq
train_file=/home/yuhai/workspace/qa/data/boolq/train.jsonl
validation_file=/home/yuhai/workspace/qa/data/boolq/dev.jsonl
load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/boolq/

epochs=20

## BERT
model_name_or_path=bert-base-uncased
cache_dir=/data/yuhai/cache/huggingface_cache/boolq_with_mix_same_in_np_boolq/bert-base-uncased
model_class=stable_cl_bert
lr=3e-5

#model_name_or_path=bert-large-uncased
#cache_dir=/data/yuhai/cache/huggingface_cache/boolq_with_mix_same_in_np_boolq/bert-large
#model_class=stable_cl_bert
#lr=1e-5

## ROBERTA
model_name_or_path=roberta-base
cache_dir=/data/yuhai/cache/huggingface_cache/boolq_with_mix_same_in_np_boolq/roberta-base
model_class=stable_cl_roberta
lr=3e-6

## Albert
#model_name_or_path=albert-base-v2
#cache_dir=/data/yuhai/cache/huggingface_cache/boolq_with_mix_same_in_np_boolq/albert-base-v2
#model_class=stable_cl_albert
#lr=1e-5

# train bs=24 18035MiB

# baseline
aux_stable_loss_weight=0.0
aux_cl_loss_weight=0.0

# stable
#aux_stable_loss_weight=0.005
#aux_cl_loss_weight=0.0

# cl
#aux_stable_loss_weight=0.0
#aux_cl_loss_weight=0.005

#stable-cl
#aux_stable_loss_weight=0.0001
#aux_cl_loss_weight=0.005

# stable-cl
#aux_stable_loss_weight=0.001
#aux_cl_loss_weight=0.001

# stable-cl
#aux_stable_loss_weight=0.05
#aux_cl_loss_weight=0.05

#export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=0

per_bs=12
aug_data_type=3   # with mixed augmented data and without dropout
aug_data_file=/home/yuhai/workspace/qa/data/np-boolq/aug_boolq_same_qa.txt
gradient_accumulation_steps=1

#per_bs=12
#aug_data_type=1
#gradient_accumulation_steps=1

#per_bs=4
#aug_data_type=1
#gradient_accumulation_steps=3

# bert-base
# train: per_bs=4, 1, 3 -> 7647MiB
# train: per_bs=12, aug_data_type=0 acc=1  -> 9791MiB
# train: per_bs=12, aug_data_type=1 acc=1  -> 15827MiB;
# bs=128 -> 4909MiB bs=256 -> 7705MiB

# bert-large
# train: per_bs=12, aug=0 acc=2 -> 23393MiB

# roberta-base
# train: per_bs=12, aug_data_type=0 acc=1  -> 10025MiB
# train: per_bs=12, aug_data_type=1 acc=1  -> 17793MiB;
# bs=128 -> MiB


output_dir=./results/"$dataset_name"/"$model_name_or_path"/"$model_class"-epochs_"$epochs"-lr_"$lr"-per_bs_"$per_bs"-aug_"$aug_data_type"-gra_acc_"$gradient_accumulation_steps"-stalw_"$aux_stable_loss_weight"-cllw_"$aux_cl_loss_weight"-$(date "+%Y-%m-%d_%H:%M:%S")

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
  --aug_data_file "$aug_data_file" \
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
  --patience 100 \
  --metric_for_best_model 'accuracy' \
  --load_best_model_at_end \
  >> "$output_dir""/log.txt" 2>&1

