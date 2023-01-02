
cd /home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/

data_list=(adv-boolq/charSwap adv-boolq/para-boolq)
validation_file_list=(/home/yuhai/workspace/qa/data/adv-boolq/charSwap/dev/val.charswap.jsonl /home/yuhai/workspace/qa/data/para-boolq/dev.jsonl)
cache_dir_list=(/data/yuhai/cache/huggingface_cache/adv-boolq/charSwap /data/yuhai/cache/huggingface_cache/adv-boolq/para-boolq)
load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/boolq/boolq.py
load_dataset_path_list=("$load_dataset_path" "$load_dataset_path")

data_1=(adv-boolq/contrast-boolq)
validation_file_1=(/home/yuhai/workspace/qa/data/contrast-boolq/boolq_perturbed.json)
cache_dir_1=(/data/yuhai/cache/huggingface_cache/adv-boolq/contrast-boolq)
load_dataset_path_1=(/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/contrast-boolq/contrast-boolq.py)

data_2=(adv-boolq/np-boolq/dev)
validation_file_2=(/home/yuhai/workspace/qa/data/np-boolq/dev.jsonl)
cache_dir_2=(/data/yuhai/cache/huggingface_cache/adv-boolq/np-boolq/dev)
load_dataset_path_2=(/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/np-boolq/np-boolq.py)

data_list=("${data_1[0]}" "${data_2[0]}")
validation_file_list=("${validation_file_1[0]}" "${validation_file_2[0]}")
cache_dir_list=("${cache_dir_1[0]}" "${cache_dir_2[0]}")
load_dataset_path_list=("${load_dataset_path_1[0]}" "${load_dataset_path_2[0]}")


# total
data_list=(adv-boolq/charSwap adv-boolq/para-boolq adv-boolq/contrast-boolq adv-boolq/np-boolq/dev)
validation_file_list=(/home/yuhai/workspace/qa/data/adv-boolq/charSwap/dev/val.charswap.jsonl /home/yuhai/workspace/qa/data/para-boolq/dev.jsonl /home/yuhai/workspace/qa/data/contrast-boolq/boolq_perturbed.json /home/yuhai/workspace/qa/data/np-boolq/dev.jsonl)
cache_dir_list=(/data/yuhai/cache/huggingface_cache/adv-boolq/charSwap /data/yuhai/cache/huggingface_cache/adv-boolq/para-boolq /data/yuhai/cache/huggingface_cache/adv-boolq/contrast-boolq /data/yuhai/cache/huggingface_cache/adv-boolq/np-boolq/dev)
load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/boolq/boolq.py
load_dataset_path_list=("$load_dataset_path" "$load_dataset_path" /home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/contrast-boolq/contrast-boolq.py /home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/np-boolq/np-boolq.py)

## BERT
model_class=stable_cl_bert
#cache_dir_list=(/data/yuhai/cache/huggingface_cache/para-boolq /data/yuhai/cache/huggingface_cache/para-np-boolq)

# baseline
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/boolq/bert-base-uncased/stable_cl_bert-epochs_20-lr_3e-5-per_bs_12-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-12_20:28:42
#model_suffix=bert-base-uncased/stable_cl_bert-epochs_20-lr_3e-5-per_bs_12-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-12_20:28:42
#
# cl
model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/boolq/bert-base-uncased/cl_bert-epochs_10-lr_3e-5-per_bs_12-aug_1-gra_acc_1-stalw_0.0-cllw_0.005-2022-05-12_13:56:37
model_suffix=bert-base-uncased/cl_bert-epochs_10-lr_3e-5-per_bs_12-aug_1-gra_acc_1-stalw_0.0-cllw_0.005-2022-05-12_13:56:37

## stable-cl
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/boolq/bert-base-uncased/stable_cl_bert-epochs_8-lr_3e-5-per_bs_12-aug_1-gra_acc_1-stalw_0.0001-cllw_0.005-2022-05-12_16:07:16
#model_suffix=bert-base-uncased/stable_cl_bert-epochs_8-lr_3e-5-per_bs_12-aug_1-gra_acc_1-stalw_0.0001-cllw_0.005-2022-05-12_16:07:16

# 0.7297
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/boolq/bert-base-uncased/stable_cl_bert-epochs_20-lr_3e-5-per_bs_12-aug_1-gra_acc_1-stalw_0.0001-cllw_0.005-2022-05-12_20:28:06
#model_suffix=bert-base-uncased/stable_cl_bert-epochs_20-lr_3e-5-per_bs_12-aug_1-gra_acc_1-stalw_0.0001-cllw_0.005-2022-05-12_20:28:06

## mix adv data
# base
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/boolq_with_mix_same_in_np_boolq/bert-base-uncased/stable_cl_bert-epochs_20-lr_3e-5-per_bs_12-aug_2-gra_acc_1-stalw_0.0-cllw_0.0-2022-06-01_00:17:21
#model_suffix=boolq_with_mix_same_in_np_boolq/bert-base-uncased/stable_cl_bert-epochs_20-lr_3e-5-per_bs_12-aug_2-gra_acc_1-stalw_0.0-cllw_0.0-2022-06-01_00:17:21
#
## sc
## 0.7596
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/boolq_with_mix_same_in_np_boolq/bert-base-uncased/stable_cl_bert-epochs_20-lr_3e-5-per_bs_12-aug_2-gra_acc_1-stalw_0.0001-cllw_0.005-2022-06-01_15:50:58
#model_suffix=boolq_with_mix_same_in_np_boolq/bert-base-uncased/stable_cl_bert-epochs_20-lr_3e-5-per_bs_12-aug_2-gra_acc_1-stalw_0.0001-cllw_0.005-2022-06-01_15:50:58

#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/boolq_with_mix_same_in_np_boolq/bert-base-uncased/stable_cl_bert-epochs_10-lr_3e-5-per_bs_12-aug_2-gra_acc_1-stalw_0.0001-cllw_0.005-2022-06-01_00:17:26
#model_suffix=boolq_with_mix_same_in_np_boolq/bert-base-uncased/stable_cl_bert-epochs_10-lr_3e-5-per_bs_12-aug_2-gra_acc_1-stalw_0.0001-cllw_0.005-2022-06-01_00:17:26

## ROBERTA
# robert-base eval-bs 128 13177MiB
#model_class=stable_cl_roberta
#cache_dir_list=(/data/yuhai/cache/huggingface_cache/para-boolq/robert-base /data/yuhai/cache/huggingface_cache/para-np-boolq/robert-base)

# baseline
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/boolq/roberta-base/stable_cl_roberta-epochs_5-lr_3e-5-per_bs_12-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-12_12:24:58
#model_suffix=roberta-base/stable_cl_roberta-epochs_5-lr_3e-5-per_bs_12-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-12_12:24:58

# aug1

#
## stable

## cl

### stable-cl
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/boolq/roberta-base/stable_cl_roberta-epochs_10-lr_3e-5-per_bs_12-aug_1-gra_acc_1-stalw_0.0001-cllw_0.005-2022-05-12_16:57:22
#model_suffix=roberta-base/stable_cl_roberta-epochs_10-lr_3e-5-per_bs_12-aug_1-gra_acc_1-stalw_0.0001-cllw_0.005-2022-05-12_16:57:22

# mix adv data
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/boolq_with_mix_same_in_np_boolq/roberta-base/stable_cl_roberta-epochs_20-lr_3e-6-per_bs_12-aug_3-gra_acc_1-stalw_0.0-cllw_0.0-2022-06-03_11:36:04
#model_suffix=boolq_with_mix_same_in_np_boolq/roberta-base/stable_cl_roberta-epochs_20-lr_3e-6-per_bs_12-aug_3-gra_acc_1-stalw_0.0-cllw_0.0-2022-06-03_11:36:04

# sc
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/boolq_with_mix_same_in_np_boolq/roberta-base/stable_cl_roberta-epochs_20-lr_3e-5-per_bs_12-aug_2-gra_acc_1-stalw_0.0001-cllw_0.005-2022-06-01_00:27:02
#model_suffix=boolq_with_mix_same_in_np_boolq/roberta-base/stable_cl_roberta-epochs_20-lr_3e-5-per_bs_12-aug_2-gra_acc_1-stalw_0.0001-cllw_0.005-2022-06-01_00:27:02

# 0.7905
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/boolq_with_mix_same_in_np_boolq/roberta-base/stable_cl_roberta-epochs_20-lr_3e-6-per_bs_12-aug_2-gra_acc_1-stalw_0.001-cllw_0.001-2022-06-03_10:35:59
#model_suffix=boolq_with_mix_same_in_np_boolq/roberta-base/stable_cl_roberta-epochs_20-lr_3e-6-per_bs_12-aug_2-gra_acc_1-stalw_0.001-cllw_0.001-2022-06-03_10:35:59

## Albert
# albert-base eval-bs 128
#model_class=stable_cl_albert

# 37
# baseline
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/boolq/albert-base-v2/stable_cl_albert-epochs_20-lr_1e-5-per_bs_12-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-18_08:38:55
#model_suffix=albert-base-v2/stable_cl_albert-epochs_20-lr_1e-5-per_bs_12-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-18_08:38:55

# aug1

#
## stable

## cl

### stable-cl
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/boolq/albert-base-v2/stable_cl_albert-epochs_20-lr_1e-5-per_bs_12-aug_1-gra_acc_1-stalw_0.0001-cllw_0.005-2022-05-18_08:50:58
#model_suffix=albert-base-v2/stable_cl_albert-epochs_20-lr_1e-5-per_bs_12-aug_1-gra_acc_1-stalw_0.0001-cllw_0.005-2022-05-18_08:50:58

# mix adv data
# base
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/boolq_with_mix_same_in_np_boolq/albert-base-v2/stable_cl_albert-epochs_20-lr_1e-5-per_bs_12-aug_2-gra_acc_1-stalw_0.0-cllw_0.0-2022-06-01_00:21:10
#model_suffix=boolq_with_mix_same_in_np_boolq/albert-base-v2/stable_cl_albert-epochs_20-lr_1e-5-per_bs_12-aug_2-gra_acc_1-stalw_0.0-cllw_0.0-2022-06-01_00:21:10
#
## sc
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/boolq_with_mix_same_in_np_boolq/albert-base-v2/stable_cl_albert-epochs_20-lr_1e-5-per_bs_12-aug_2-gra_acc_1-stalw_0.0001-cllw_0.005-2022-06-01_00:24:59
#model_suffix=boolq_with_mix_same_in_np_boolq/albert-base-v2/stable_cl_albert-epochs_20-lr_1e-5-per_bs_12-aug_2-gra_acc_1-stalw_0.0001-cllw_0.005-2022-06-01_00:24:59

export CUDA_VISIBLE_DEVICES=0
export python=/home/yuhai/workspace/anaconda3/envs/hug4.16/bin/python

for i in {0..3}
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
      --per_device_eval_batch_size 64 \
      --cache_dir "$cache_dir" \
      --output_dir "$output_dir" \
      >> "$output_dir""/log.txt" 2>&1
done