
cd /home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/

data_para=adv-squad-v2/para_ques
para_file=/home/yuhai/workspace/qa/data/adv_squad_v2/para_ques/paraphrase_dev_by_ques_gen_with_1_q.json
para_cache=/data/yuhai/cache/huggingface_cache/adv-squad-v2/para_ques
##
#data_list=(adv-squad-v2/adv_doc)
#eval_file_list=(/home/yuhai/workspace/qa/data/adv_squad_v2/adv_doc/dev-v2.0.addsent.json)
#cache_dir=/data/yuhai/cache/huggingface_cache/adv-squad-v2/adv_doc

data_addonesent=adv-squad-v2/adv_doc_add_one_sent
addonesent_file=/home/yuhai/workspace/qa/data/squad_v2/dev-v2.0.add_one_sent.json
addonesent_cache=/data/yuhai/cache/huggingface_cache/adv-squad-v2/adv_doc_add_one_sent

data_list=(adv-squad-v2/charSwap "$data_para" "$data_addonesent")
eval_file_list=(/home/yuhai/workspace/qa/data/adv-squad-v2.0/charSwap/dev/dev.charswap.json "$para_file" "$addonesent_file")
cache_dir_list=(/data/yuhai/cache/huggingface_cache/adv-squad-v2.0/charSwap "$para_cache" "$addonesent_cache")

load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/squad_v2/

#model_class=baseline_bert
#model_class=stable_bert
#model_class=cl_bert
model_class=stable_cl_bert

## bert-base version
# base
model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/bert-base-uncased/squad_v2-baseline_bert-bs_12-aug_data_type_0-2022-03-17_19:28:44
model_suffix=bert-base-uncased/squad_v2-baseline_bert-bs_12-aug_data_type_0-2022-03-17_19:28:44

# stable-cl
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/bert-base-uncased/stable_cl_bert-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-26_17:09:18
#model_suffix=bert-base-uncased/stable_cl_bert-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-26_17:09:18

## ROBERTA
#model_class=baseline_roberta
#model_class=stable_roberta
#model_class=cl_roberta
#model_class=stable_cl_roberta
#
## roberta base
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/roberta-base/baseline_roberta-bs_12-aug_data_type_0-2022-03-22_10:49:44
#model_suffix=roberta-base/baseline_roberta-bs_12-aug_data_type_0-2022-03-22_10:49:44-2

# sc
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/roberta-base/stable_cl_roberta-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-21_11:02:17
#model_suffix=roberta-base/stable_cl_roberta-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-21_11:02:17


#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad/roberta-large/baseline_roberta-bs_12-aug_data_type_0-2022-03-15_09:18:46
#model_suffix=roberta-large/baseline_roberta-bs_12-aug_data_type_0-2022-03-15_09:18:46

# albert
model_class=stable_cl_albert

#
## baseline
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/albert-base-v2/stable_cl_albert-per_bs_12-aug_0-gacc_1-stalw_0.0-cllw_0.0-2022-05-18_13:29:52
#model_suffix=albert-base-v2/stable_cl_albert-per_bs_12-aug_0-gacc_1-stalw_0.0-cllw_0.0-2022-05-18_13:29:52

# 1e-5
model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/albert-base-v2/stable_cl_albert-lr-1e-5-per_bs_12-aug_0-gacc_1-stalw_0.0-cllw_0.0-2022-06-05_15:07:55
model_suffix=albert-base-v2/stable_cl_albert-lr-1e-5-per_bs_12-aug_0-gacc_1-stalw_0.0-cllw_0.0-2022-06-05_15:07:55

# sc
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/albert-base-v2/stable_cl_albert-per_bs_12-aug_1-gacc_1-stalw_0.0001-cllw_0.005-2022-05-18_19:21:39
#model_suffix=albert-base-v2/stable_cl_albert-per_bs_12-aug_1-gacc_1-stalw_0.0001-cllw_0.005-2022-05-18_19:21:39

#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/albert-base-v2/stable_cl_albert-per_bs_12-aug_1-gacc_1-stalw_0.001-cllw_0.001-2022-06-05_00:53:40
#model_suffix=albert-base-v2/stable_cl_albert-per_bs_12-aug_1-gacc_1-stalw_0.001-cllw_0.001-2022-06-05_00:53:40

export CUDA_VISIBLE_DEVICES=4
export python=/home/yuhai/workspace/anaconda3/envs/hug4.16/bin/python

for i in {0..2}
  do
    data=${data_list["$i"]}
    eval_file=${eval_file_list["$i"]}
    cache_dir=${cache_dir_list["$i"]}/"$model_class"
    output_dir=./results/"$data"/"$model_suffix"

    if [ ! -d "$output_dir"  ];then
      mkdir -p "$output_dir"
    else
      echo "$output_dir" exist
    fi

    "$python" ./run_qa.py \
      --model_name_or_path "$model_name_or_path"  \
      --model_class "$model_class" \
      --validation_file "$eval_file" \
      --version_2_with_negative \
      --load_dataset_path "$load_dataset_path" \
      --do_eval \
      --per_device_eval_batch_size 256 \
      --output_dir "$output_dir" \
      >> "$output_dir""/log.txt" 2>&1 \
      --cache_dir "$cache_dir"
done
