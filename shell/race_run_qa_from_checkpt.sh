cd /home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/

dataset_name=race/all/dev
train_folder=/home/yuhai/workspace/qa/data/race/train/**
validation_folder=/home/yuhai/workspace/qa/data/race/dev/**
load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/race/

## BERT
model_name_or_path=bert-base-uncased
cache_dir=/data/yuhai/cache/huggingface_cache/race/

#model_name_or_path=bert-large-uncased
#cache_dir=/data/yuhai/cache/huggingface_cache/race/bert-large
#
model_class=stable_cl_bert
lr=3e-5
epochs=4
checkpoint=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/dev/bert-base-uncased/stable_cl_bert-lr_3e-5-per_bs_3-aug_1-gra_acc_2-stalw_0.005-cllw_0.0005-2022-05-11_10:22:36
cpt_name=stable_cl_bert-lr_3e-5-per_bs_3-aug_1-gra_acc_2-stalw_0.005-cllw_0.0005-2022-05-11_10:22:36


## ROBERTA
#model_name_or_path=roberta-base     # bs=6 6949 bs=12 10110 bs=24 16065
##model_name_or_path=roberta-large    # bs=8 17371MiB; bs=12 23663MiB
#cache_dir=/data/yuhai/cache/huggingface_cache/race/roberta-base
#
#model_class=stable_cl_roberta
#lr=3e-6

# baseline
#aux_stable_loss_weight=0.0
#aux_cl_loss_weight=0.0

# stable
#aux_stable_loss_weight=0.005
#aux_cl_loss_weight=0.0

# cl
#aux_stable_loss_weight=0.0
#aux_cl_loss_weight=0.005

# stable-cl
aux_stable_loss_weight=0.005
aux_cl_loss_weight=0.5

# stable-cl
#aux_stable_loss_weight=0.05
#aux_cl_loss_weight=0.05

#export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=2

#per_bs=6    # 6 examples => 6 * num_choices = 24 instances
#aug_data_type=0
#gradient_accumulation_steps=1

per_bs=3    # 6 examples => 6 * num_choices = 24 instances
aug_data_type=1
gradient_accumulation_steps=2


# train: per_bs=4, aug_data_type=0 acc=1  -> 11103MiB
# train: per_bs=4, aug_data_type=1 acc=1  -> 19277MiB
# train: per_bs=6, aug_data_type=0 acc=1  ->
# train: per_bs=6, aug_data_type=1 acc=1  -> OOV;
# eval: bs=32 -> 4395MiB; bs=128 -> 12603MiB

# batch_size=24 instance (1 instance is quesiton and context with 1 option)
output_dir=./results/"$dataset_name"/"$model_name_or_path"/epochs_"$epochs"-start_from-checkpt-"$cpt_name"-$(date "+%Y-%m-%d_%H:%M:%S")

if [ ! -d "$output_dir"  ];then
  mkdir -p "$output_dir"
else
  echo "$output_dir" exist
fi

export python=/home/yuhai/workspace/anaconda3/envs/hug4.16/bin/python
"$python" ./run_mc_qa.py \
  --model_name_or_path "$model_name_or_path"  \
  --resume_from_checkpoint "$checkpoint" \
  --ignore_data_skip \
  --model_class "$model_class" \
  --aux_stable_loss_weight "$aux_stable_loss_weight" \
  --aux_cl_loss_weight "$aux_cl_loss_weight" \
  --train_folder "$train_folder" \
  --name "all" \
  --save_strategy 'epoch' \
  --validation_folder "$validation_folder" \
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
  >> "$output_dir""/log.txt" 2>&1

