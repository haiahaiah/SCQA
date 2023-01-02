cd /home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/

dataset_name=race/all/dev
train_folder=/home/yuhai/workspace/qa/data/race/train/**
validation_folder=/home/yuhai/workspace/qa/data/race/dev/**
load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/race/

epochs=10
max_steps=12000
### Albert
model_name=albert-large-v2
model_name_or_path=albert-large-v2
cache_dir=/data/yuhai/cache/huggingface_cache/race/albert-large-v2

# albert-base
# train: bs=2 6155MiB; bs=4 10363MiB; bs=6 14669MiB; bs=12 oov;
# eval: bs=128 ok;

# albert-large
# train: bs=2 19109MiB;
# eval: bs=64 15307MiB;

# albert-xxlarge
# train: bs=1 oov

model_class=stable_cl_albert
lr=2e-5

# orig
type_name=orig
aux_stable_loss_weight=0.0
aux_cl_loss_weight=0.0

# stable
#type_name=stable
#aux_stable_loss_weight=0.0001
#aux_cl_loss_weight=0.0

# cl
#type_name=cl
#aux_stable_loss_weight=0.0
#aux_cl_loss_weight=0.005

# stable then cl
#epochs=3
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/dev/albert-base-v2/stable-epochs_3-lr_1e-5-per_bs_3-aug_1-gra_acc_2-stalw_0.0001-cllw_0.0-2022-06-10_22:05:19
#type_name=stable_then_cl
#aux_stable_loss_weight=0.0
#aux_cl_loss_weight=0.005

# cl then stable
#epochs=3
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/dev/albert-base-v2/cl-epochs_3-lr_1e-5-per_bs_3-aug_1-gra_acc_2-stalw_0.0-cllw_0.005-2022-06-10_22:05:50
#type_name=cl_then_stable
#aux_stable_loss_weight=0.0001
#aux_cl_loss_weight=0.0

## stable-cl
#epochs=3
#type_name=stable_cl
#aux_stable_loss_weight=0.005
#aux_cl_loss_weight=0.0005


export CUDA_VISIBLE_DEVICES=0,1,2,3

seed=42
#seed=2022

per_bs=2    # 6 examples => 6 * num_choices = 24 instances
aug_data_type=0
gradient_accumulation_steps=4
gpu=4

#per_bs=1    # 6 examples => 6 * num_choices = 24 instances
#aug_data_type=1
#gradient_accumulation_steps=6

output_dir=./results/"$dataset_name"/"$model_name"/"$type_name"/lr_"$lr"/seed_"$seed"-per_bs_"$per_bs"-aug_"$aug_data_type"-gra_acc_"$gradient_accumulation_steps"-gpu_"$gpu"-stalw_"$aux_stable_loss_weight"-cllw_"$aux_cl_loss_weight"-$(date "+%Y-%m-%d_%H:%M:%S")

if [ ! -d "$output_dir"  ];then
  mkdir -p "$output_dir"
else
  echo "$output_dir" exist
fi

export python=/home/yuhai/workspace/anaconda3/envs/hug4.16/bin/python
"$python" ./run_mc_qa.py \
  --model_name_or_path "$model_name_or_path"  \
  --model_class "$model_class" \
  --aux_stable_loss_weight "$aux_stable_loss_weight" \
  --aux_cl_loss_weight "$aux_cl_loss_weight" \
  --train_folder "$train_folder" \
  --name "all" \
  --validation_folder "$validation_folder" \
  --load_dataset_path "$load_dataset_path" \
  --cache_dir "$cache_dir" \
  --aug_data_type "$aug_data_type" \
  --do_train \
  --seed "$seed" \
  --do_eval \
  --per_device_train_batch_size "$per_bs" \
  --gradient_accumulation_steps "$gradient_accumulation_steps" \
  --per_device_eval_batch_size 64 \
  --learning_rate "$lr" \
  --warmup_steps 1000 \
  --num_train_epochs "$epochs" \
  --max_steps "$max_steps" \
  --max_seq_length 512 \
  --output_dir "$output_dir" \
  >> "$output_dir""/log.txt" 2>&1 \
  --metric_for_best_model 'accuracy' \
  --load_best_model_at_end \
  --fp16 \
  --evaluation_strategy 'steps' \
  --eval_steps 1000 \
  --save_steps 1000 \
  --save_total_limit 3