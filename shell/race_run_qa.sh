cd /home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/

dataset_name=race/all/dev
train_folder=/home/yuhai/workspace/qa/data/race/train/**
validation_folder=/home/yuhai/workspace/qa/data/race/dev/**
load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/race/

epochs=6

## BERT
model_class=stable_cl_bert

# bert-base
# bs=6 13607 MiB; bs=12 23571MiB
# bert-large
# bs=3 20453MiB
# bs=4 oov

model_name=bert-base-uncased
model_name_or_path=bert-base-uncased
cache_dir=/data/yuhai/cache/huggingface_cache/race/bert
lr=3e-5

#model_name=bert-large-uncased
#model_name_or_path=bert-large-uncased
#cache_dir=/data/yuhai/cache/huggingface_cache/race/bert-large
#lr=1e-5

## ROBERTA
# model_name=roberta-base
#model_name_or_path=roberta-base     # bs=6 6949 bs=12 10110 bs=24 16065
#cache_dir=/data/yuhai/cache/huggingface_cache/race/roberta-base
##
#model_class=stable_cl_roberta
#lr=3e-6

### Albert
#model_name_or_path=albert-base-v2
#cache_dir=/data/yuhai/cache/huggingface_cache/race/albert-base-v2
#model_class=stable_cl_albert
#lr=1e-5

#model_name_or_path=albert-xxlarge-v2
#cache_dir=/data/yuhai/cache/huggingface_cache/race/albert-xxlarge-v2
#model_class=stable_cl_albert
#lr=1e-5

# baseline
type_name=orig
aux_stable_loss_weight=0.0
aux_cl_loss_weight=0.0

#type_name=drop
#aux_stable_loss_weight=0.0
#aux_cl_loss_weight=0.0

# stable
#type_name=stable
#aux_stable_loss_weight=0.005
#aux_cl_loss_weight=0.0

# cl
#type_name=cl
#aux_stable_loss_weight=0.0
#aux_cl_loss_weight=0.0005

# stable then cl
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/dev/bert-base-uncased/stable-epochs_3-lr_3e-5-per_bs_6-aug_1-gra_acc_1-stalw_0.005-cllw_0.0-2022-06-09_19:42:16
#type_name=stable_then_cl
#aux_stable_loss_weight=0.0
#aux_cl_loss_weight=0.0005

# cl then stable
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/dev/bert-base-uncased/cl_bert-epochs_3-lr_3e-5-per_bs_6-aug_1-gra_acc_1-stalw_0.0-cllw_0.0005-2022-06-09_19:37:50
#type_name=cl_then_stable
#aux_stable_loss_weight=0.005
#aux_cl_loss_weight=0.0

## stable-cl
#type_name=stable_cl
#aux_stable_loss_weight=0.005
#aux_cl_loss_weight=0.0005

#export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=3
#seed=625 # epoch=6: 0.6534
#seed=829 # epoch=6: no;
#seed=1017  # no
#seed=1234 # no
#seed=1016  # 0.5836
#seed=2022  # 0.6644

#seed=1231  # no
#seed=409   # 0.6585
#seed=1997  # 0.6568
#seed=2019   # 0.6699


per_bs=6    # 6 examples => 6 * num_choices = 24 instances
aug_data_type=0
gradient_accumulation_steps=1

#per_bs=6    # 6 examples => 6 * num_choices = 24 instances
#aug_data_type=1
#gradient_accumulation_steps=1
#
#per_bs=3    # 6 examples => 6 * num_choices = 24 instances
#aug_data_type=1
#gradient_accumulation_steps=2

# batch_size=24 instance (1 instance is quesiton and context with 1 option)
output_dir=./results/"$dataset_name"/"$model_name"/"$type_name"/epochs_"$epochs"/lr_"$lr"/seed_"$seed"-per_bs_"$per_bs"-aug_"$aug_data_type"-gra_acc_"$gradient_accumulation_steps"-stalw_"$aux_stable_loss_weight"-cllw_"$aux_cl_loss_weight"-$(date "+%Y-%m-%d_%H:%M:%S")

if [ ! -d "$output_dir"  ];then
  mkdir -p "$output_dir"
else
  echo "$output_dir" exist
fi

export python=/home/yuhai/workspace/anaconda3/envs/hug4.16/bin/python
"$python" ./run_mc_qa.py \
  --model_name_or_path "$model_name_or_path"  \
  --model_class "$model_class" \
  --seed "$seed" \
  --aux_stable_loss_weight "$aux_stable_loss_weight" \
  --aux_cl_loss_weight "$aux_cl_loss_weight" \
  --train_folder "$train_folder" \
  --name "all" \
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
  >> "$output_dir""/log.txt" 2>&1 \
  --metric_for_best_model 'accuracy' \
  --load_best_model_at_end \
  --fp16 \
  --evaluation_strategy 'epoch' \
  --save_strategy "epoch" \
  --save_total_limit 3 \
  --patience 3 \
#  --evaluation_strategy 'steps' \
#  --eval_steps 2000 \
#  --save_steps 2000 \
#  --save_total_limit 20 \
#  --patience 5 \

