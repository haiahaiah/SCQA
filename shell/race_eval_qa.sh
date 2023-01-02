cd /home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/

load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/race/

data_list=(race/all/dev race/all/test/ race/high/dev/ race/middle/dev/ race/high/test/ race/middle/test/)
validation_folder_list=(/home/yuhai/workspace/qa/data/race/dev /home/yuhai/workspace/qa/data/race/test /home/yuhai/workspace/qa/data/race/dev/high/ /home/yuhai/workspace/qa/data/race/dev/middle/ /home/yuhai/workspace/qa/data/race/test/high/ /home/yuhai/workspace/qa/data/race/test/middle/)
cache_dir_list=(/data/yuhai/cache/huggingface_cache/race/dev/all /data/yuhai/cache/huggingface_cache/race/test/all /data/yuhai/cache/huggingface_cache/para-race/test/all /data/yuhai/cache/huggingface_cache/race/dev/high /data/yuhai/cache/huggingface_cache/race/dev/middle /data/yuhai/cache/huggingface_cache/race/test/high /data/yuhai/cache/huggingface_cache/race/test/middle)

## BERT
model_class=stable_cl_bert
#
## baseline
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/bert-base-uncased/stable_cl_bert-per_bs_6-aug_data_type_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-07_21:36:35
#model_suffix=bert-base-uncased/stable_cl_bert-per_bs_6-aug_data_type_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-07_21:36:35
model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/dev/bert-base-uncased/stable_cl_bert-epochs_3-lr_3e-5-per_bs_6-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-24_23:58:13
model_suffix=bert-base-uncased/stable_cl_bert-epochs_3-lr_3e-5-per_bs_6-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-24_23:58:13

# aug1
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/bert-base-uncased/stable_cl_bert-per_bs_3-aug_data_type_1-gra_acc_2-stalw_0.0-cllw_0.0-2022-05-07_21:47:15
#model_suffix=bert-base-uncased/stable_cl_bert-per_bs_3-aug_data_type_1-gra_acc_2-stalw_0.0-cllw_0.0-2022-05-07_21:47:15

# stable
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/bert-base-uncased/stable_cl_bert-per_bs_3-aug_data_type_1-gra_acc_2-stalw_0.005-cllw_0.0-2022-05-09_01:47:48
#model_suffix=bert-base-uncased/stable_cl_bert-per_bs_3-aug_data_type_1-gra_acc_2-stalw_0.005-cllw_0.0-2022-05-09_01:47:48

# cl
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/bert-base-uncased/stable_cl_bert-per_bs_3-aug_data_type_1-gra_acc_2-stalw_0.0-cllw_0.005-2022-05-07_22:25:54
#model_suffix=bert-base-uncased/stable_cl_bert-per_bs_3-aug_data_type_1-gra_acc_2-stalw_0.0-cllw_0.005-2022-05-07_22:25:54

# stable-cl
#model_name_or_path=model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/dev/bert-base-uncased/stable_cl_bert-epochs_4-lr_3e-5-per_bs_3-aug_1-gra_acc_3-stalw_0.001-cllw_0.001-2022-05-12_10:22:22
#model_suffix=bert-base-uncased/stable_cl_bert-epochs_4-lr_3e-5-per_bs_3-aug_1-gra_acc_3-stalw_0.001-cllw_0.001-2022-05-12_10:22:22

## ROBERTA
# roberta-base eval-bs 128 13177MiB
# roberta-base eval-bs 64  7705MiB
#model_class=stable_cl_roberta
#cache_dir=/data/yuhai/cache/huggingface_cache/race/roberta-base
#
#model_folder=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/dev/roberta-base/
# baseline
#model_name_or_path="$model_folder"/stable_cl_roberta-lr_3e-6-per_bs_6-aug_data_type_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-10_00:11:46
#model_suffix=roberta-base/stable_cl_roberta-lr_3e-6-per_bs_6-aug_data_type_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-10_00:11:46

# aug1
#model_name_or_path="$model_folder"/stable_cl_roberta-lr_3e-6-per_bs_3-aug_1-gra_acc_2-stalw_0.0-cllw_0.0-2022-05-10_01:34:10
#model_suffix=roberta-base/stable_cl_roberta-lr_3e-6-per_bs_3-aug_1-gra_acc_2-stalw_0.0-cllw_0.0-2022-05-10_01:34:10

# stable
#model_name_or_path="$model_folder"/stable_cl_roberta-lr_3e-6-per_bs_3-aug_1-gra_acc_2-stalw_0.005-cllw_0.0-2022-05-10_01:35:00
#model_suffix=roberta-base/stable_cl_roberta-lr_3e-6-per_bs_3-aug_1-gra_acc_2-stalw_0.005-cllw_0.0-2022-05-10_01:35:00
#
## cl
#model_name_or_path="$model_folder"/stable_cl_roberta-lr_3e-6-per_bs_3-aug_1-gra_acc_2-stalw_0.0-cllw_0.005-2022-05-10_10:22:51
#model_suffix=roberta-base/stable_cl_roberta-lr_3e-6-per_bs_3-aug_1-gra_acc_2-stalw_0.0-cllw_0.005-2022-05-10_10:22:51
##
### stable-cl
#model_name_or_path="$model_folder"/stable_cl_roberta-lr_3e-6-per_bs_3-aug_1-gra_acc_2-stalw_0.005-cllw_0.005-2022-05-10_01:34:39
#model_suffix=roberta-base/stable_cl_roberta-lr_3e-6-per_bs_3-aug_1-gra_acc_2-stalw_0.005-cllw_0.005-2022-05-10_01:34:39

# baseline
#aux_stable_loss_weight=0.0
#aux_cl_loss_weight=0.0

# stable-cl
#aux_stable_loss_weight=0.005
#aux_cl_loss_weight=0.005

export CUDA_VISIBLE_DEVICES=1
export python=/home/yuhai/workspace/anaconda3/envs/hug4.16/bin/python

for i in {1..5}
  do
    data=${data_list["$i"]}
    validation_folder=${validation_folder_list["$i"]}
    cache_dir=${cache_dir_list["$i"]}/"$model_class"
    output_dir=./results/"$data"/"$model_suffix"
    if [ ! -d "$output_dir"  ];then
      mkdir -p "$output_dir"
    else
      echo "$output_dir" exist
    fi

    "$python" ./run_mc_qa.py \
      --model_name_or_path "$model_name_or_path"  \
      --model_class "$model_class" \
      --validation_folder "$validation_folder" \
      --load_dataset_path "$load_dataset_path" \
      --do_eval \
      --per_device_eval_batch_size 64 \
      --cache_dir "$cache_dir" \
      --output_dir "$output_dir" \
      >> "$output_dir""/log.txt" 2>&1
done