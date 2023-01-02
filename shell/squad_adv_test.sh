
cd /home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/

data_list=(Para-SQuAD/adv_orig Para-SQuAD/adv_para Para-SQuAD/dev_orig Para-SQuAD/dev_para Para-SQuAD/dev_orig_by_quesgen squad_adv_emnlp2017/add_sent squad_adv_emnlp2017/add_one_sent adv-squad-v1.1/charSwap SQuAD)
squad_folder=/home/yuhai/workspace/qa/data/SQuAD
para_squad_folder=/home/yuhai/workspace/qa/data/Para-SQuAD
squad_adv_folder=/home/yuhai/workspace/qa/data/squad-adversarial
eval_file_list=("$para_squad_folder"/adv_orig.json "$para_squad_folder"/adv_para.json "$para_squad_folder"/dev_orig.json "$para_squad_folder"/dev_para.json "$para_squad_folder"/paraphrase_dev_1094_by_ques_gen_with_1_q.json "$squad_adv_folder"/add_sent.json "$squad_adv_folder"/add_one_sent.json /home/yuhai/workspace/qa/data/adv-squad-v1.1/charSwap/dev/dev.charswap.json "$squad_folder"/dev-v1.1.json)

load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/squad/

#model_class=baseline_bert
#model_class=stable_bert
#model_class=cl_bert
model_class=stable_cl_bert


## bert-base version

model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad/bert-base-uncased/stable_cl_bert-seed_42-epoch_3-lr_1e-5-per_bs_6-aug_data_type_1-gc_2-stalw_0.0-cllw_0.0-2022-08-28_17:18:34/checkpoint-17000
model_suffix=bert-base-uncased/stable_cl_bert-seed_42-epoch_3-lr_1e-5-per_bs_6-aug_data_type_1-gc_2-stalw_0.0-cllw_0.0-2022-08-28_17:18:34/checkpoint-17000

#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad/bert-base-uncased/stable_cl_bert-epoch_3-lr_2e-5-per_bs_12-aug_data_type_0-gc_1-stalw_0.0-cllw_0.0-2022-05-28_09:38:18
#model_suffix=bert-base-uncased/stable_cl_bert-epoch_3-lr_2e-5-per_bs_12-aug_data_type_0-gc_1-stalw_0.0-cllw_0.0-2022-05-28_09:38:18

#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad/bert-base-uncased/stable_cl_bert-epoch_3-lr_1e-5-per_bs_12-aug_data_type_0-gc_1-stalw_0.0-cllw_0.0-2022-08-28_15:05:29
#model_suffix=bert-base-uncased/stable_cl_bert-epoch_3-lr_1e-5-per_bs_12-aug_data_type_0-gc_1-stalw_0.0-cllw_0.0-2022-08-28_15:05:29

#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad/bert-base-uncased/stable_cl_bert-seed_2019-epoch_3-lr_1e-5-per_bs_12-aug_data_type_0-gc_1-stalw_0.0-cllw_0.0-2022-08-28_15:26:55
#model_suffix=bert-base-uncased/stable_cl_bert-seed_2019-epoch_3-lr_1e-5-per_bs_12-aug_data_type_0-gc_1-stalw_0.0-cllw_0.0-2022-08-28_15:26:55

#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_with_mix_diverse_20/bert-base-uncased/stable_cl_bert-epoch_3-lr_3e-5-per_bs_12-aug_0-gacc_1-stalw_0.0-cllw_0.0-2022-05-26_10:25:47/checkpoint-22000
#model_suffix=squad_with_mix_diverse_20/bert-base-uncased/stable_cl_bert-epoch_3-lr_3e-5-per_bs_12-aug_0-gacc_1-stalw_0.0-cllw_0.0-2022-05-26_10:25:47/checkpoint-22000
#216
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_with_mix_diverse_20/bert-base-uncased/stable_cl_bert-epoch_3-lr_3e-5-per_bs_12-aug_2-gacc_1-stalw_0.005-cllw_0.005-2022-05-26_13:40:12/checkpoint-15000
#model_suffix=squad_with_mix_diverse_20/bert-base-uncased/stable_cl_bert-epoch_3-lr_3e-5-per_bs_12-aug_2-gacc_1-stalw_0.005-cllw_0.005-2022-05-26_13:40:12/checkpoint-15000


#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad/bert-base-uncased/stable_cl_bert-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-26_17:36:45
#model_suffix=bert-base-uncased/stable_cl_bert-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-26_17:36:45

# mix
# base
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_with_mix_diverse_20/bert-base-uncased/stable_cl_bert-epoch_3-lr_3e-5-per_bs_12-aug_0-gacc_1-stalw_0.0-cllw_0.0-2022-05-26_10:25:47
#model_suffix=squad_with_mix_diverse_20/bert-base-uncased/stable_cl_bert-epoch_3-lr_3e-5-per_bs_12-aug_0-gacc_1-stalw_0.0-cllw_0.0-2022-05-26_10:25:47

# sc 216
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_with_mix_diverse_20/bert-base-uncased/stable_cl_bert-epoch_3-lr_3e-5-per_bs_12-aug_2-gacc_1-stalw_0.005-cllw_0.005-2022-05-26_13:40:12
#model_suffix=squad_with_mix_diverse_20/bert-base-uncased/stable_cl_bert-epoch_3-lr_3e-5-per_bs_12-aug_2-gacc_1-stalw_0.005-cllw_0.005-2022-05-26_13:40:12

## bert-large version

## ROBERTA
#model_class=baseline_roberta
#model_class=stable_roberta
#model_class=cl_roberta
#model_class=stable_cl_roberta
#
## roberta base
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad/roberta-base/stable_cl_roberta-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-26_17:37:08/checkpoint-15000
#model_suffix=roberta-base/stable_cl_roberta-bs_24-aug_data_type_1-stalw_0.005-cllw_0.005-2022-04-26_17:37:08/checkpoint-15000

# sc
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_with_mix_diverse_20/roberta-base/stable_cl_roberta-epoch_3-lr_3e-5-per_bs_12-aug_2-gacc_1-stalw_0.005-cllw_0.005-2022-05-26_13:43:32
#model_suffix=squad_with_mix_diverse_20/roberta-base/stable_cl_roberta-epoch_3-lr_3e-5-per_bs_12-aug_2-gacc_1-stalw_0.005-cllw_0.005-2022-05-26_13:43:32

#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_with_mix_diverse_20/roberta-base/stable_cl_roberta-epoch_3-lr_3e-5-per_bs_12-aug_0-gacc_1-stalw_0.0-cllw_0.0-2022-05-31_10:28:12/checkpoint-18500
#model_suffix=squad_with_mix_diverse_20/roberta-base/stable_cl_roberta-epoch_3-lr_3e-5-per_bs_12-aug_0-gacc_1-stalw_0.0-cllw_0.0-2022-05-31_10:28:12/checkpoint-18500
#
## sc 122
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_with_mix_diverse_20/roberta-base/stable_cl_roberta-epoch_3-lr_3e-5-per_bs_12-aug_2-gacc_1-stalw_0.005-cllw_0.005-2022-05-31_14:34:50/checkpoint-18500
#model_suffix=squad_with_mix_diverse_20/roberta-base/stable_cl_roberta-epoch_3-lr_3e-5-per_bs_12-aug_2-gacc_1-stalw_0.005-cllw_0.005-2022-05-31_14:34:50/checkpoint-18500

# mix

# 216 sc
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_with_mix_diverse_20/roberta-base/stable_cl_roberta-epoch_3-lr_3e-5-per_bs_12-aug_2-gacc_1-stalw_0.005-cllw_0.005-2022-05-26_13:43:32
#model_suffix=squad_with_mix_diverse_20/roberta-base/stable_cl_roberta-epoch_3-lr_3e-5-per_bs_12-aug_2-gacc_1-stalw_0.005-cllw_0.005-2022-05-26_13:43:32


#model_class=stable_cl_albert
##

#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/SQuAD/albert-base-v2/stable_cl_albert-per_bs_12-aug_0-gacc_1-stalw_0.0-cllw_0.0-2022-05-18_09:32:05/checkpoint-20000
#model_suffix=albert-base-v2/stable_cl_albert-per_bs_12-aug_0-gacc_1-stalw_0.0-cllw_0.0-2022-05-18_09:32:05/checkpoint-20000

#
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad/albert-base-v2/stable_cl_albert-epoch_5-lr_1e-5-per_bs_12-aug_1-gacc_1-stalw_0.0001-cllw_0.005-2022-05-19_14:57:09
#model_suffix=albert-base-v2/stable_cl_albert-epoch_5-lr_1e-5-per_bs_12-aug_1-gacc_1-stalw_0.0001-cllw_0.005-2022-05-19_14:57:09

## mix base
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_with_mix_diverse_20/albert-base-v2/stable_cl_albert-epoch_3-lr_3e-5-per_bs_12-aug_0-gacc_1-stalw_0.0-cllw_0.0-2022-05-26_16:23:44
#model_suffix=squad_with_mix_diverse_20/albert-base-v2/stable_cl_albert-epoch_3-lr_3e-5-per_bs_12-aug_0-gacc_1-stalw_0.0-cllw_0.0-2022-05-26_16:23:44
### mix sc
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_with_mix_diverse_20/albert-base-v2/stable_cl_albert-epoch_3-lr_3e-5-per_bs_12-aug_2-gacc_1-stalw_0.005-cllw_0.005-2022-05-26_13:44:00
#model_suffix=squad_with_mix_diverse_20/albert-base-v2/stable_cl_albert-epoch_3-lr_3e-5-per_bs_12-aug_2-gacc_1-stalw_0.005-cllw_0.005-2022-05-26_13:44:00

#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad/albert-base-v2/stable_cl_albert-epoch_3-lr_1e-5-per_bs_12-aug_data_type_1-gc_1-stalw_0.001-cllw_0.001-2022-06-05_15:16:40
#model_suffix=albert-base-v2/stable_cl_albert-epoch_3-lr_1e-5-per_bs_12-aug_data_type_1-gc_1-stalw_0.001-cllw_0.001-2022-06-05_15:16:40

cache_dir=/data/yuhai/cache/huggingface_cache

export CUDA_VISIBLE_DEVICES=2
export python=/home/yuhai/workspace/anaconda3/envs/hug4.16/bin/python

for i in {5..6}
  do
    data=${data_list["$i"]}
    eval_file=${eval_file_list["$i"]}
    cache_dir="$cache_dir"/"$data"/"$model_class"
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
      --per_device_eval_batch_size 64 \
      --cache_dir "$cache_dir" \
      --output_dir "$output_dir" \
      >> "$output_dir""/log.txt" 2>&1
done

# eval bs=8 2397MiB
