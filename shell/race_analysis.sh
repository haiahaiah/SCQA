cd /home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/

load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/race/

#data_list=(para-race/all/dev/ para-race/all/test/ para-race/high/dev/ para-race/middle/dev/ para-race/high/test/ para-race/middle/test/)
#validation_folder_list=(/home/yuhai/workspace/qa/data/para-race/dev/ /home/yuhai/workspace/qa/data/para-race/test /home/yuhai/workspace/qa/data/para-race/dev/high/ /home/yuhai/workspace/qa/data/para-race/dev/middle/ /home/yuhai/workspace/qa/data/para-race/test/high/ /home/yuhai/workspace/qa/data/para-race/test/middle/)
#cache_dir_list=(/data/yuhai/cache/huggingface_cache/para-race/dev/all/ /data/yuhai/cache/huggingface_cache/para-race/test/all/ /data/yuhai/cache/huggingface_cache/para-race/dev/high /data/yuhai/cache/huggingface_cache/para-race/dev/middle /data/yuhai/cache/huggingface_cache/para-race/test/high /data/yuhai/cache/huggingface_cache/para-race/test/middle)

data_list=(adv-race/test/orig/all adv-race/test/addsent/all adv-race/test/charswap/all adv-race/test/para-ques/all  adv-race/test/de/all adv-race/test/dg/all)
orig_race_folder=/home/yuhai/workspace/qa/data/adv-race-test-folder
adv_race_folder=/home/yuhai/workspace/qa/data/adv-race
validation_folder_list=("$orig_race_folder"/Orig "$adv_race_folder"/AddSent "$adv_race_folder"/charSwap /home/yuhai/workspace/qa/data/para-race/test "$adv_race_folder"/DE "$adv_race_folder"/DG)
cache_folder=/data/yuhai/cache/huggingface_cache/adv-race-test-folder-2
cache_dir_list=("$cache_folder"/Orig "$cache_folder"/AddSent "$cache_folder"/charSwap "$cache_folder"/para-ques "$cache_folder"/DE "$cache_folder"/DG)

## BERT
model_class=stable_cl_bert
#
## baseline
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/dev/bert-base-uncased/stable_cl_bert-epochs_3-lr_3e-5-per_bs_6-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-24_23:58:13
#model_suffix=bert-base-uncased/stable_cl_bert-epochs_3-lr_3e-5-per_bs_6-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-24_23:58:13

# aug1
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/bert-base-uncased/stable_cl_bert-per_bs_3-aug_data_type_1-gra_acc_2-stalw_0.0-cllw_0.0-2022-05-07_21:47:15
#model_suffix=bert-base-uncased/stable_cl_bert-per_bs_3-aug_data_type_1-gra_acc_2-stalw_0.0-cllw_0.0-2022-05-07_21:47:15

# stable
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/dev/bert-base-uncased/stable-epochs_3-lr_3e-5-per_bs_6-aug_1-gra_acc_1-stalw_0.005-cllw_0.0-2022-06-09_19:42:16
#model_suffix=bert-base-uncased/stable-epochs_3-lr_3e-5-per_bs_6-aug_1-gra_acc_1-stalw_0.005-cllw_0.0-2022-06-09_19:42:16

# cl
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/dev/bert-base-uncased/cl_bert-epochs_3-lr_3e-5-per_bs_6-aug_1-gra_acc_1-stalw_0.0-cllw_0.0005-2022-06-09_19:37:50
#model_suffix=bert-base-uncased/cl_bert-epochs_3-lr_3e-5-per_bs_6-aug_1-gra_acc_1-stalw_0.0-cllw_0.0005-2022-06-09_19:37:50

# stable-cl
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/dev/bert-base-uncased/stable_cl_bert-epochs_4-lr_3e-5-per_bs_3-aug_1-gra_acc_3-stalw_0.001-cllw_0.001-2022-05-12_10:22:22
#model_suffix=bert-base-uncased/stable_cl_bert-epochs_4-lr_3e-5-per_bs_3-aug_1-gra_acc_3-stalw_0.001-cllw_0.001-2022-05-12_10:22:22

# 0.68
model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/dev/bert-base-uncased/stable_cl_bert-lr_3e-5-per_bs_3-aug_1-gra_acc_2-stalw_0.005-cllw_0.0005-2022-05-11_10:22:36
model_suffix=bert-base-uncased/stable_cl_bert-lr_3e-5-per_bs_3-aug_1-gra_acc_2-stalw_0.005-cllw_0.0005-2022-05-11_10:22:36


## ROBERTA
# robert-base eval-bs 128 13177MiB
#model_class=stable_cl_roberta
##
### baseline
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/dev/roberta-base/stable_cl_roberta-lr_3e-6-per_bs_6-aug_data_type_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-10_00:11:46
#model_suffix=roberta-base/stable_cl_roberta-lr_3e-6-per_bs_6-aug_data_type_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-10_00:11:46
# aug1
#model_name_or_path="$model_folder"/stable_cl_roberta-lr_3e-6-per_bs_3-aug_1-gra_acc_2-stalw_0.0-cllw_0.0-2022-05-10_01:34:10
#model_suffix=roberta-base/stable_cl_roberta-lr_3e-6-per_bs_3-aug_1-gra_acc_2-stalw_0.0-cllw_0.0-2022-05-10_01:34:10
#
## stable
#model_name_or_path="$model_folder"/stable_cl_roberta-lr_3e-6-per_bs_3-aug_1-gra_acc_2-stalw_0.005-cllw_0.0-2022-05-10_01:35:00
#model_suffix=roberta-base/stable_cl_roberta-lr_3e-6-per_bs_3-aug_1-gra_acc_2-stalw_0.005-cllw_0.0-2022-05-10_01:35:00
#
## cl
#model_name_or_path="$model_folder"/stable_cl_roberta-lr_3e-6-per_bs_3-aug_1-gra_acc_2-stalw_0.0-cllw_0.005-2022-05-10_10:22:51
#model_suffix=roberta-base/stable_cl_roberta-lr_3e-6-per_bs_3-aug_1-gra_acc_2-stalw_0.0-cllw_0.005-2022-05-10_10:22:51
##
### stable-cl
# dev 0.7266
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/dev/roberta-base/stable_cl_roberta-lr_3e-6-per_bs_3-aug_1-gra_acc_2-stalw_0.005-cllw_0.005-2022-05-10_01:34:39
#model_suffix=roberta-base/stable_cl_roberta-lr_3e-6-per_bs_3-aug_1-gra_acc_2-stalw_0.005-cllw_0.005-2022-05-10_01:34:39
# dev 0.7254
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/dev/roberta-base/stable_cl_roberta-lr_3e-6-per_bs_3-aug_1-gra_acc_2-stalw_0.0005-cllw_0.0005-2022-05-12_00:10:40
#model_suffix=roberta-base/stable_cl_roberta-lr_3e-6-per_bs_3-aug_1-gra_acc_2-stalw_0.0005-cllw_0.0005-2022-05-12_00:10:40

model_class=stable_cl_albert
#### base
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/dev/albert-base-v2/stable_cl_albert-epochs_10-lr_1e-6-per_bs_6-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-18_19:28:26
#model_suffix=albert-base-v2/stable_cl_albert-epochs_10-lr_1e-6-per_bs_6-aug_0-gra_acc_1-stalw_0.0-cllw_0.0-2022-05-18_19:28:26
##
# sc 122
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/dev/albert-base-v2/stable_cl_albert-epochs_10-lr_1e-6-per_bs_3-aug_1-gra_acc_2-stalw_0.0001-cllw_0.005-2022-05-18_19:28:39
#model_suffix=albert-base-v2/stable_cl_albert-epochs_10-lr_1e-6-per_bs_3-aug_1-gra_acc_2-stalw_0.0001-cllw_0.005-2022-05-18_19:28:39

## 216 linux
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/dev/albert-base-v2/stable_cl_albert-epochs_8-lr_1e-5-per_bs_3-aug_1-gra_acc_2-stalw_0.0001-cllw_0.005-2022-05-18_13:34:02
#model_suffix=albert-base-v2/stable_cl_albert-epochs_8-lr_1e-5-per_bs_3-aug_1-gra_acc_2-stalw_0.0001-cllw_0.005-2022-05-18_13:34:02

# stable cl
model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/dev/albert-base-v2/stable_cl/epochs_3/lr_1e-5/per_bs_1-aug_1-gra_acc_6-stalw_0.005-cllw_0.0005-2022-06-13_01:55:00
model_suffix=albert-base-v2/stable_cl/epochs_3/lr_1e-5/per_bs_1-aug_1-gra_acc_6-stalw_0.005-cllw_0.0005-2022-06-13_01:55:00

export CUDA_VISIBLE_DEVICES=1
export python=/home/yuhai/workspace/anaconda3/envs/hug4.16/bin/python

# todo
# para-ques like adv-race format or change reading adv-data code
#validation_folder=${validation_folder_list["$0"]}
validation_folder=/home/yuhai/workspace/qa/data/adv-race-test-folder/Orig
for i in {1..2}
  do
    data=${data_list["$i"]}
    adv_file=${validation_folder_list["$i"]}/test_dis.json
    cache_dir=${cache_dir_list["$i"]}/"$model_class"
    output_dir=./results/"$data"/"$model_suffix"
    if [ ! -d "$output_dir"  ];then
      mkdir -p "$output_dir"
    else
      echo "$output_dir" exist
    fi

    "$python" ./eval_mc_qa_align_uniform.py \
      --model_name_or_path "$model_name_or_path"  \
      --model_class "$model_class" \
      --validation_folder "$validation_folder" \
      --load_dataset_path "$load_dataset_path" \
      --do_sent_emb \
      --dropout \
      --adv_file "$adv_file" \
      --adv_type 1 \
      --per_device_eval_batch_size 32 \
      --cache_dir "$cache_dir" \
      --output_dir "$output_dir" \
      >> "$output_dir""/log.txt" 2>&1
done
