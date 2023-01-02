
cd /home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/

dataset_name=reclor
train_file=/home/yuhai/workspace/qa/data/reclor/train.json
validation_file=/home/yuhai/workspace/qa/data/reclor/val.json
load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/reclor/

epochs=20

## BERT
model_name_or_path=bert-base-uncased
cache_dir=/data/yuhai/cache/huggingface_cache/reclor/bert-base-uncased

#model_name_or_path=bert-large-uncased
#cache_dir=/data/yuhai/cache/huggingface_cache/reclor/bert-large
#
model_class=stable_cl_bert
lr=3e-5

## ROBERTA
#model_name_or_path=roberta-base     # bs=6 6949 bs=12 10110 bs=24 16065
#cache_dir=/data/yuhai/cache/huggingface_cache/reclor/roberta-base
##
#model_class=stable_cl_roberta
#lr=3e-6

# baseline
aux_stable_loss_weight=0.0
aux_cl_loss_weight=0.0

# stable
#aux_stable_loss_weight=0.0005
#aux_cl_loss_weight=0.0

# cl
#aux_stable_loss_weight=0.0
#aux_cl_loss_weight=0.0005

# stable-cl
#aux_stable_loss_weight=0.0001
#aux_cl_loss_weight=0.005

# stable-cl
#aux_stable_loss_weight=0.05
#aux_cl_loss_weight=0.05

#export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=0

per_bs=6    # 6 examples => 6 * num_choices = 24 instances
aug_data_type=0
gradient_accumulation_steps=1

#per_bs=3    # 6 examples => 6 * num_choices = 24 instances
#aug_data_type=1
#gradient_accumulation_steps=2


# train: per_bs=1, aug_data_type=0 acc=6  -> 5591MiB
# train: per_bs=4, aug_data_type=1 acc=1  ->
# train: per_bs=6, aug_data_type=0 acc=1  ->
# train: per_bs=6, aug_data_type=1 acc=1  -> ;
# eval: bs=32 -> ; bs=128 ->

# batch_size=24 instance (1 instance is quesiton and context with 1 option)
output_dir=./results/"$dataset_name"/"$model_name_or_path"/"$model_class"-epochs_"$epochs"-lr_"$lr"-per_bs_"$per_bs"-aug_"$aug_data_type"-gra_acc_"$gradient_accumulation_steps"-stalw_"$aux_stable_loss_weight"-cllw_"$aux_cl_loss_weight"-$(date "+%Y-%m-%d_%H:%M:%S")

if [ ! -d "$output_dir"  ];then
  mkdir -p "$output_dir"
else
  echo "$output_dir" exist
fi

export python=/home/yuhai/workspace/anaconda3/envs/hug4.16/bin/python
"$python" ./run_reclor_qa.py \
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
  --per_device_eval_batch_size 32 \
  --learning_rate "$lr" \
  --num_train_epochs "$epochs" \
  --evaluation_strategy 'epoch' \
  --save_strategy "epoch" \
  --save_total_limit 2 \
  --patience 20 \
  --metric_for_best_model 'accuracy' \
  --load_best_model_at_end \
  --output_dir "$output_dir" \
  >> "$output_dir""/log.txt" 2>&1

