cd /home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/

dataset_name=race/all/dev
train_folder=/home/yuhai/workspace/qa/data/race/train/**
validation_folder=/home/yuhai/workspace/qa/data/race/dev/**
load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/race/

epochs=3
### Albert
model_name=albert-base-v2

# albert-base bs=2 6155MiB; bs=4 10363MiB; bs=6 14669MiB; bs=12 oov;

model_name_or_path=albert-base-v2
cache_dir=/data/yuhai/cache/huggingface_cache/race/albert-base-v2
model_class=stable_cl_albert
lr=1e-5

#model_name_or_path=albert-xxlarge-v2
#cache_dir=/data/yuhai/cache/huggingface_cache/race/albert-xxlarge-v2
#model_class=stable_cl_albert
#lr=1e-5

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
epochs=3
type_name=stable_cl
aux_stable_loss_weight=0.005
aux_cl_loss_weight=0.0005


export CUDA_VISIBLE_DEVICES=3
#per_bs=6    # 6 examples => 6 * num_choices = 24 instances
#aug_data_type=1
#gradient_accumulation_steps=1

per_bs=1    # 6 examples => 6 * num_choices = 24 instances
aug_data_type=1
gradient_accumulation_steps=6

# batch_size=24 instance (1 instance is quesiton and context with 1 option)
output_dir=./results/"$dataset_name"/"$model_name"/"$type_name"/epochs_"$epochs"/lr_"$lr"/per_bs_"$per_bs"-aug_"$aug_data_type"-gra_acc_"$gradient_accumulation_steps"-stalw_"$aux_stable_loss_weight"-cllw_"$aux_cl_loss_weight"-$(date "+%Y-%m-%d_%H:%M:%S")

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
