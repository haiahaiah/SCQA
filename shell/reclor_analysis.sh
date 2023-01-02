cd /home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/

load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/reclor/

data_list=(reclor/orig-dev/ adv-reclor/charswap_q+context/ adv-reclor/para-ques/ adv-reclor/addsent/)
reclor_folder=/home/yuhai/workspace/qa/data/reclor/dev/
adv_reclor_folder=/home/yuhai/workspace/qa/data/adv-reclor
validation_folder_list=("$reclor_folder" "$adv_reclor_folder"/charSwap/dev/attack_q+context "$adv_reclor_folder"/para-reclor/dev "$adv_reclor_folder"/addsent/)
cache_folder=/data/yuhai/cache/huggingface_cache/adv-reclor
cache_dir_list=(/data/yuhai/cache/huggingface_cache/reclor "$cache_folder"/charSwap_q+context "$cache_folder"/para-reclor "$cache_folder"/addsent-reclor)

## BERT
model_class=stable_cl_bert

# baseline
model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/reclor/bert-base-uncased/stable_cl_bert-epochs_30-lr_2e-5-per_bs_6-aug_0-gra_acc_4-stalw_0.0-cllw_0.0-2022-05-15_10:57:36
model_suffix=bert-base-uncased/stable_cl_bert-epochs_30-lr_2e-5-per_bs_6-aug_0-gra_acc_4-stalw_0.0-cllw_0.0-2022-05-15_10:57:36

# stable-cl
model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/reclor/bert-base-uncased/stable_cl_bert-epochs_40-lr_2e-5-per_bs_3-aug_1-gra_acc_8-stalw_0.0001-cllw_0.005-2022-05-18_00:15:58
model_suffix=bert-base-uncased/stable_cl_bert-epochs_40-lr_2e-5-per_bs_3-aug_1-gra_acc_8-stalw_0.0001-cllw_0.005-2022-05-18_00:15:58
#
### ROBERTA
## robert-base eval-bs 128 13177MiB
model_class=stable_cl_roberta
#
##model_folder=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/race/all/dev/roberta-base/
## baseline
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/reclor/roberta-base/stable_cl_roberta-epochs_40-lr_1e-5-per_bs_3-aug_0-gra_acc_8-stalw_0.0-cllw_0.0-2022-05-18_00:17:09
#model_suffix=roberta-base/stable_cl_roberta-epochs_40-lr_1e-5-per_bs_3-aug_0-gra_acc_8-stalw_0.0-cllw_0.0-2022-05-18_00:17:09

# base
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/reclor/roberta-base/stable_cl_roberta-epochs_15-lr_2e-5-per_bs_12-aug_0-gra_acc_2-stalw_0.0-cllw_0.0-2022-06-03_11:39:27
#model_suffix=roberta-base/stable_cl_roberta-epochs_15-lr_2e-5-per_bs_12-aug_0-gra_acc_2-stalw_0.0-cllw_0.0-2022-06-03_11:39:27

# 0.538
model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/reclor/roberta-base/stable_cl_roberta-epochs_15-lr_2e-5-per_bs_12-aug_0-gra_acc_2-stalw_0.0-cllw_0.0-2022-06-03_11:39:27
model_suffix=roberta-base/stable_cl_roberta-epochs_15-lr_2e-5-per_bs_12-aug_0-gra_acc_2-stalw_0.0-cllw_0.0-2022-06-03_11:39:27

# sc
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/reclor/roberta-base/stable_cl_roberta-epochs_20-lr_2e-5-per_bs_8-aug_1-gra_acc_3-stalw_0.005-cllw_0.0001-2022-06-03_10:20:40
#model_suffix=roberta-base/stable_cl_roberta-epochs_20-lr_2e-5-per_bs_8-aug_1-gra_acc_3-stalw_0.005-cllw_0.0001-2022-06-03_10:20:40

### stable-cl
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/reclor/roberta-base/stable_cl_roberta-epochs_40-lr_1e-5-per_bs_3-aug_1-gra_acc_8-stalw_0.0001-cllw_0.005-2022-05-17_15:49:07
#model_suffix=roberta-base/stable_cl_roberta-epochs_40-lr_1e-5-per_bs_3-aug_1-gra_acc_8-stalw_0.0001-cllw_0.005-2022-05-17_15:49:07

#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/reclor/roberta-base/stable_cl_roberta-epochs_20-lr_2e-5-per_bs_8-aug_1-gra_acc_3-stalw_0.0001-cllw_0.005-2022-06-02_21:15:29
#model_suffix=roberta-base/stable_cl_roberta-epochs_20-lr_2e-5-per_bs_8-aug_1-gra_acc_3-stalw_0.0001-cllw_0.005-2022-06-02_21:15:29

# 0.544
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/reclor/roberta-base/stable_cl_roberta-epochs_20-lr_2e-5-per_bs_8-aug_1-gra_acc_3-stalw_0.0001-cllw_0.005-2022-06-02_21:15:29
#model_suffix=roberta-base/stable_cl_roberta-epochs_20-lr_2e-5-per_bs_8-aug_1-gra_acc_3-stalw_0.0001-cllw_0.005-2022-06-02_21:15:29

#0.558
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/reclor/roberta-base/stable_cl_roberta-epochs_20-lr_3e-5-per_bs_12-aug_0-gra_acc_2-stalw_0.0-cllw_0.0-2022-06-04_17:30:50
#model_suffix=roberta-base/stable_cl_roberta-epochs_20-lr_3e-5-per_bs_12-aug_0-gra_acc_2-stalw_0.0-cllw_0.0-2022-06-04_17:30:50

#model_class=stable_cl_albert
####
#0.534
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/reclor/albert-base-v2/stable_cl_albert-epochs_40-lr_1e-5-per_bs_6-aug_0-gra_acc_4-stalw_0.0-cllw_0.0-2022-05-18_12:19:35
#model_suffix=albert-base-v2/stable_cl_albert-epochs_40-lr_1e-5-per_bs_6-aug_0-gra_acc_4-stalw_0.0-cllw_0.0-2022-05-18_12:19:35
# 0.57
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/reclor/albert-base-v2/stable_cl_albert-epochs_20-lr_2e-5-per_bs_8-aug_0-gra_acc_3-stalw_0.0-cllw_0.0-2022-05-25_12:59:47
#model_suffix=albert-base-v2/stable_cl_albert-epochs_20-lr_2e-5-per_bs_8-aug_0-gra_acc_3-stalw_0.0-cllw_0.0-2022-05-25_12:59:47
#
#
## stable-cl
# 0.55
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/reclor/albert-base-v2/stable_cl_albert-epochs_20-lr_1e-5-per_bs_8-aug_1-gra_acc_3-stalw_0.001-cllw_0.001-2022-05-25_09:38:34
#model_suffix=albert-base-v2/stable_cl_albert-epochs_20-lr_1e-5-per_bs_8-aug_1-gra_acc_3-stalw_0.001-cllw_0.001-2022-05-25_09:38:34

#0.58
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/reclor/albert-base-v2/stable_cl_albert-epochs_20-lr_2e-5-per_bs_8-aug_1-gra_acc_3-stalw_0.005-cllw_0.0001-2022-05-25_09:36:44
#model_suffix=albert-base-v2/stable_cl_albert-epochs_20-lr_2e-5-per_bs_8-aug_1-gra_acc_3-stalw_0.005-cllw_0.0001-2022-05-25_09:36:44

export CUDA_VISIBLE_DEVICES=1
export python=/home/yuhai/workspace/anaconda3/envs/hug4.16/bin/python

validation_folder=/home/yuhai/workspace/qa/data/reclor/dev
for i in {1..3}
  do
    data=${data_list["$i"]}
    adv_file=${validation_folder_list["$i"]}/val.json
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