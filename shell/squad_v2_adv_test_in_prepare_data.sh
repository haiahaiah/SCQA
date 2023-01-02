
cd /home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/

#data_list=(adv-squad-v2/para_ques)
#eval_file_list=(/home/yuhai/workspace/qa/data/adv_squad_v2/para_ques/paraphrase_dev_by_ques_gen_with_1_q.json)

data_list=(adv-squad-v2/adv_doc_addsent)
#eval_file_list=(/home/yuhai/workspace/qa/data/adv_squad_v2/adv_doc/dev-v2.0.addsent.json)
eval_file_list=(/home/yuhai/workspace/qa/data/squad_v2/dev-v2.0.json)

adv_type=2
adv_file=/home/yuhai/workspace/qa/data/squad_v2/dev-v2.0.addsent_data.txt

load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/squad_v2/

#model_class=baseline_bert
#model_class=stable_bert
#model_class=cl_bert
model_class=stable_cl_bert


## bert-base version
model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/bert-base-uncased/squad_v2-baseline_bert-bs_12-aug_data_type_0-2022-03-17_19:28:44
model_suffix=bert-base-uncased/squad_v2-baseline_bert-bs_12-aug_data_type_0-2022-03-17_19:28:44

#216 linux
model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/bert-base-uncased/stable_cl_bert-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-26_10:08:28
model_suffix=bert-base-uncased/stable_cl_bert-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-26_10:08:28

#122 linux
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/bert-base-uncased/stable_cl_bert-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-26_17:09:18
#model_suffix=bert-base-uncased/stable_cl_bert-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-26_17:09:18

#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/bert-base-uncased/stable_cl_bert-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-26_10:08:28
#model_suffix=bert-base-uncased/stable_cl_bert-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-26_10:08:28

#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/bert-base-uncased/stable_cl_bert-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-26_17:09:18
#model_suffix=bert-base-uncased/stable_cl_bert-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-26_17:09:18

## ROBERTA
#model_class=baseline_roberta
#model_class=stable_roberta
#model_class=cl_roberta
#model_class=stable_cl_roberta

# roberta base

#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/roberta-base/baseline_roberta-bs_12-aug_data_type_0-2022-03-22_10:49:44
#model_suffix=roberta-base/baseline_roberta-bs_12-aug_data_type_0-2022-03-22_10:49:44
#
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/roberta-base/stable_cl_roberta-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-21_11:02:17
#model_suffix=roberta-base/stable_cl_roberta-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-21_11:02:17

## roberta-large

#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad/roberta-large/baseline_roberta-bs_12-aug_data_type_0-2022-03-15_09:18:46
#model_suffix=roberta-large/baseline_roberta-bs_12-aug_data_type_0-2022-03-15_09:18:46

cache_dir=/data/yuhai/cache/huggingface_cache
export CUDA_VISIBLE_DEVICES=2
export python=/home/yuhai/workspace/anaconda3/envs/hug4.16/bin/python

for i in {0..0}
  do
    data=${data_list["$i"]}
    eval_file=${eval_file_list["$i"]}
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
      --overwrite_cache \
      --adv_type "$adv_type" \
      --adv_file "$adv_file" \
      --per_device_eval_batch_size 128 \
      --cache_dir "$cache_dir" \
      --output_dir "$output_dir" \
      >> "$output_dir""/log.txt" 2>&1
done
