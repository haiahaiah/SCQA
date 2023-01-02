
cd /home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/


dataset_name=squad_with_mix_diverse_20
train_file=/home/yuhai/workspace/qa/data/squad-addsent-diverse/train-mixed-AddSentDiverse-20.json
validation_file=/home/yuhai/workspace/qa/data/SQuAD/dev-v1.1.json
cache_dir=/data/yuhai/cache/huggingface_cache/squad-addsent-diverse
load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/squad/

lr=1e-5
epoch=3

## BERT
model_name_or_path=bert-base-uncased   # bs=24 15297MiB
#model_name_or_path=bert-large-uncased

#model_class=baseline_bert
#model_class=cl_bert
#model_class=stable_bert
model_class=stable_cl_bert

## ROBERTA
model_name_or_path=roberta-base      # bs=24 15573MiB
#model_name_or_path=roberta-large    # bs=8 17371MiB;

#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad/roberta-base/stable_cl_roberta-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-26_17:37:08

#model_class=baseline_roberta
#model_class=stable_roberta
#model_class=cl_roberta
model_class=stable_cl_roberta

## Albert
model_name_or_path=albert-base-v2
model_class=stable_cl_albert
#epoch=5
# train bs=24 14765MiB

# baseline
aux_stable_loss_weight=0.0
aux_cl_loss_weight=0.0

# stable-cl
#aux_stable_loss_weight=0.0001
#aux_cl_loss_weight=0.005

#aux_stable_loss_weight=0.005
#aux_cl_loss_weight=0.005

#export CUDA_VISIBLE_DEVICES=3,2,1,0
export CUDA_VISIBLE_DEVICES=0
#NUM_GPU=1
per_bs=12
aug_data_type=0
gradient_accumulation_steps=1

#per_bs=12
#aug_data_type=2   # with mixed augmented data and dropout
#aug_data_file=/home/yuhai/workspace/qa/data/squad-addsent-diverse/qid2aug_contexts.json
#gradient_accumulation_steps=1

# batch_size=12
# batch_size=24
output_dir=./results/"$dataset_name"/"$model_name_or_path"/"$model_class"-epoch_"$epoch"-lr_"$lr"-per_bs_"$per_bs"-aug_"$aug_data_type"-gacc_"$gradient_accumulation_steps"-stalw_"$aux_stable_loss_weight"-cllw_"$aux_cl_loss_weight"-$(date "+%Y-%m-%d_%H:%M:%S")

if [ ! -d "$output_dir"  ];then
  mkdir -p "$output_dir"
else
  echo "$output_dir" exist
fi

export python=/home/yuhai/workspace/anaconda3/envs/hug4.16/bin/python
"$python" ./run_qa.py \
  --model_name_or_path "$model_name_or_path"  \
  --model_class "$model_class" \
  --aux_stable_loss_weight "$aux_stable_loss_weight" \
  --aux_cl_loss_weight "$aux_cl_loss_weight" \
  --train_file "$train_file" \
  --validation_file "$validation_file" \
  --load_dataset_path "$load_dataset_path" \
  --cache_dir "$cache_dir"/"$model_class" \
  --do_train \
  --aug_data_type "$aug_data_type" \
  --do_eval \
  --per_device_train_batch_size "$per_bs" \
  --gradient_accumulation_steps "$gradient_accumulation_steps" \
  --per_device_eval_batch_size 64 \
  --learning_rate "$lr" \
  --num_train_epochs "$epoch" \
  --output_dir "$output_dir" \
  >> "$output_dir""/log.txt" 2>&1 \
  --fp16 \
