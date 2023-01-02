

cd /home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/

#boolq_load_dataset=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/boolq/boolq.py
#np_load_dataset=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/np-boolq/np-boolq.py
#load_dataset_path_list=("$np_load_dataset" "$np_load_dataset")
#data_list=(np-boolq/all np-boolq/dev)
#validation_file_list=(/home/yuhai/workspace/qa/data/np-boolq/all.jsonl /home/yuhai/workspace/qa/data/np-boolq/dev.jsonl)

#load_dataset_path_list=("$boolq_load_dataset" "$np_load_dataset")
#data_list=(para-boolq para-np-boolq)
#validation_file_list=(/home/yuhai/workspace/qa/data/para-boolq/dev.jsonl /home/yuhai/workspace/qa/data/para-np-boolq/dev.jsonl)

data_list=(adv-np-boolq/charSwap adv-np-boolq/para-ques adv-np-boolq/contrast-boolq adv-np-boolq/boolq-dev)
validation_file_list=(/home/yuhai/workspace/qa/data/adv-np-boolq/charSwap/dev/val.charswap.jsonl /home/yuhai/workspace/qa/data/para-np-boolq/dev.jsonl /home/yuhai/workspace/qa/data/contrast-boolq/boolq_perturbed.json /home/yuhai/workspace/qa/data/boolq/dev.jsonl)
cache_dir_list=(/data/yuhai/cache/huggingface_cache/adv-np-boolq/charSwap /data/yuhai/cache/huggingface_cache/adv-np-boolq/para-ques /data/yuhai/cache/huggingface_cache/adv-np-boolq/contrast-boolq /data/yuhai/cache/huggingface_cache/adv-np-boolq/boolq-dev)
boolq_load_dataset=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/boolq/
np_load_dataset=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/np-boolq/np-boolq.py
contrast_load_dataset=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/contrast-boolq/contrast-boolq.py
load_dataset_path_list=("$np_load_dataset" "$np_load_dataset" "$contrast_load_dataset" "$boolq_load_dataset")


## BERT
model_class=stable_cl_bert
#cache_dir_list=(/data/yuhai/cache/huggingface_cache/para-boolq /data/yuhai/cache/huggingface_cache/para-np-boolq)

# baseline
model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/np-boolq/bert-base-uncased/stable_cl_bert-epochs_20-lr_3e-5-per_bs_12-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-13_01:46:23
model_suffix=bert-base-uncased/stable_cl_bert-epochs_20-lr_3e-5-per_bs_12-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-13_01:46:23

# aug1

# stable

# cl

# stable-cl
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/np-boolq/bert-base-uncased/stable_cl_bert-epochs_20-lr_3e-5-per_bs_6-aug_1-gra_acc_2-stalw_0.0001-cllw_0.005-2022-05-13_01:48:31
#model_suffix=bert-base-uncased/stable_cl_bert-epochs_20-lr_3e-5-per_bs_6-aug_1-gra_acc_2-stalw_0.0001-cllw_0.005-2022-05-13_01:48:31

## ROBERTA
# robert-base eval-bs 128 13177MiB
#model_class=stable_cl_roberta
#cache_dir_list=(/data/yuhai/cache/huggingface_cache/para-boolq/robert-base /data/yuhai/cache/huggingface_cache/para-np-boolq/robert-base)

# baseline
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/np-boolq/roberta-base/stable_cl_roberta-epochs_20-lr_3e-6-per_bs_12-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-13_10:16:31
#model_suffix=roberta-base/stable_cl_roberta-epochs_20-lr_3e-6-per_bs_12-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-13_10:16:31
# aug1

#
## stable

## cl

### stable-cl
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/np-boolq/roberta-base/stable_cl_roberta-epochs_40-patience_30-lr_3e-6-per_bs_6-aug_1-gra_acc_2-stalw_0.0001-cllw_0.005-2022-05-13_22:47:43
#model_suffix=roberta-base/stable_cl_roberta-epochs_40-patience_30-lr_3e-6-per_bs_6-aug_1-gra_acc_2-stalw_0.0001-cllw_0.005-2022-05-13_22:47:43

# albert

model_class=stable_cl_albert

# base
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/np-boolq/albert-base-v2/stable_cl_albert-epochs_20-patience_10-lr_2e-5-per_bs_12-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-26_00:31:33
#model_suffix=stable_cl_albert-epochs_20-patience_10-lr_2e-5-per_bs_12-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-26_00:31:33

# 0.7054
model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/np-boolq/albert-base-v2/stable_cl_albert-epochs_40-patience_30-lr_1e-5-per_bs_12-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-18_08:58:24
model_suffix=albert-base-v2/stable_cl_albert-epochs_40-patience_30-lr_1e-5-per_bs_12-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-18_08:58:24

# sc
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/np-boolq/albert-base-v2/stable_cl_albert-epochs_20-patience_10-lr_2e-5-per_bs_12-aug_1-gra_acc_1-stalw_0.0001-cllw_0.005-2022-05-26_00:31:47
#model_suffix=stable_cl_albert-epochs_20-patience_10-lr_2e-5-per_bs_12-aug_1-gra_acc_1-stalw_0.0001-cllw_0.005-2022-05-26_00:31:47

# 37 0.7077
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/np-boolq/albert-base-v2/stable_cl_albert-epochs_40-patience_30-lr_1e-5-per_bs_12-aug_1-gra_acc_1-stalw_0.0001-cllw_0.005-2022-05-18_13:05:42
#model_suffix=albert-base-v2/stable_cl_albert-epochs_40-patience_30-lr_1e-5-per_bs_12-aug_1-gra_acc_1-stalw_0.0001-cllw_0.005-2022-05-18_13:05:42

export CUDA_VISIBLE_DEVICES=0
export python=/home/yuhai/workspace/anaconda3/envs/hug4.16/bin/python

for i in {3..3}
  do
    data=${data_list["$i"]}
    load_dataset_path=${load_dataset_path_list["$i"]}
    validation_file=${validation_file_list["$i"]}
    cache_dir=${cache_dir_list["$i"]}/"$model_class"
    output_dir=./results/"$data"/"$model_suffix"
    if [ ! -d "$output_dir"  ];then
      mkdir -p "$output_dir"
    else
      echo "$output_dir" exist
    fi

    "$python" ./run_yn_qa.py \
      --model_name_or_path "$model_name_or_path"  \
      --model_class "$model_class" \
      --validation_file "$validation_file" \
      --load_dataset_path "$load_dataset_path" \
      --do_eval \
      --per_device_eval_batch_size 128 \
      --cache_dir "$cache_dir" \
      --output_dir "$output_dir" \
      >> "$output_dir""/log.txt" 2>&1
done