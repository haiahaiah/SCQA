
cd /home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/

data_list=(Para-SQuAD/adv_orig Para-SQuAD/adv_para Para-SQuAD/dev_orig Para-SQuAD/dev_para Para-SQuAD/dev_orig_by_quesgen squad_adv_emnlp2017/add_sent squad_adv_emnlp2017/add_one_sent SQuAD)
squad_folder=/home/yuhai/workspace/qa/data/SQuAD
para_squad_folder=/home/yuhai/workspace/qa/data/Para-SQuAD
squad_adv_folder=/home/yuhai/workspace/qa/data/squad-adversarial
eval_file_list=("$para_squad_folder"/adv_orig.json "$para_squad_folder"/adv_para.json "$para_squad_folder"/dev_orig.json "$para_squad_folder"/dev_para.json "$para_squad_folder"/paraphrase_dev_1094_by_ques_gen_with_1_q.json "$squad_adv_folder"/add_sent.json "$squad_adv_folder"/add_one_sent.json "$squad_folder"/dev-v1.1.json)


data_list=(adv-squad-v1.1/charSwap)
eval_file_list=(/home/yuhai/workspace/qa/data/adv-squad-v1.1/charSwap/dev/dev.charswap.json)
load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/squad/
cache_dir_list=(/data/yuhai/cache/huggingface_cache/adv-squad-v1.1/charSwap)

#model_class=baseline_bert
#model_class=stable_bert
#model_class=cl_bert
model_class=stable_cl_bert

## bert-base version
# base
model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad/bert-base-uncased/baseline-bs_12-aug_data_type_0-2022-03-10_15:32:17
model_suffix=bert-base-uncased/baseline-bs_12-aug_data_type_0-2022-03-10_15:32:17

## stable-cl
model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad/bert-base-uncased/stable_cl_bert-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-26_17:36:45
model_suffix=bert-base-uncased/stable_cl_bert-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-26_17:36:45


## ROBERTA
#model_class=baseline_roberta
#model_class=stable_roberta
#model_class=cl_roberta
model_class=stable_cl_roberta

# roberta base
# baseline
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad/roberta-base/baseline_roberta-bs_12-aug_data_type_0-2022-03-11_18:29:01
#model_suffix=roberta-base/baseline_roberta-bs_12-aug_data_type_0-2022-03-11_18:29:01

## stable-cl
model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad/roberta-base/stable_cl_roberta-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-26_17:37:08
model_suffix=roberta-base/stable_cl_roberta-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-26_17:37:08

model_class=stable_cl_albert
#cac_dir=/data/yuhai/cache/huggingface_cache/squad-1.1/albert-base-v2
#cache_dir_list=("$cac_dir"/adv_orig "$cac_dir"/adv_para "$cac_dir"/dev_orig "$cac_dir"/dev_para "$cac_dir"/adv_orig "$cac_dir"/para_by_ques_gen "$cac_dir"/add_sent "$cac_dir"/add_one_sent "$cac_dir"/squad-dev)

# baseline
model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad/albert-base-v2/stable_cl_albert-per_bs_12-aug_0-gacc_1-stalw_0.0-cllw_0.0-2022-05-18_09:32:05
model_suffix=albert-base-v2/stable_cl_albert-per_bs_12-aug_0-gacc_1-stalw_0.0-cllw_0.0-2022-05-18_09:32:05

# sc
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad/albert-base-v2/stable_cl_albert-per_bs_12-aug_1-gacc_1-stalw_0.0001-cllw_0.005-2022-05-18_09:38:47
#model_suffix=albert-base-v2/stable_cl_albert-per_bs_12-aug_1-gacc_1-stalw_0.0001-cllw_0.005-2022-05-18_09:38:47

export CUDA_VISIBLE_DEVICES=2
export python=/home/yuhai/workspace/anaconda3/envs/hug4.16/bin/python

for i in {0..0}
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
      --load_dataset_path "$load_dataset_path" \
      --do_eval \
      --per_device_eval_batch_size 128 \
      --cache_dir "$cache_dir" \
      --output_dir "$output_dir" \
      >> "$output_dir""/log.txt" 2>&1
done
