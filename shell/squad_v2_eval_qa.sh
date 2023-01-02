
cd /home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/

data_list=(squad-v2/)
eval_file_list=(/home/yuhai/workspace/qa/data/squad_v2/dev-v2.0.json)

load_dataset_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/datasets/squad_v2/

#model_class=stable_cl_bert

## bert-base version

## ROBERTA
#model_class=stable_cl_roberta

# roberta base

#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/roberta-base/cl_roberta-bs_24-aug_data_type_1-stalw_0-cllw_0.005-2022-04-26_16:54:14
#model_suffix=roberta-base/cl_roberta-bs_24-aug_data_type_1-stalw_0-cllw_0.005-2022-04-26_16:54:14

# albert-base-v2
model_class=stable_cl_albert
cache_dir=/data/yuhai/cache/huggingface_cache/squad-2.0/albert-base-v2/

# baseline
model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/albert-base-v2/stable_cl_albert-per_bs_12-aug_0-gacc_1-stalw_0.0-cllw_0.0-2022-05-18_13:29:52
model_suffix=albert-base-v2/stable_cl_albert-per_bs_12-aug_0-gacc_1-stalw_0.0-cllw_0.0-2022-05-18_13:29:52

# 1e-5
model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/albert-base-v2/stable_cl_albert-lr-1e-5-per_bs_12-aug_0-gacc_1-stalw_0.0-cllw_0.0-2022-06-05_15:07:55
model_suffix=albert-base-v2/stable_cl_albert-lr-1e-5-per_bs_12-aug_0-gacc_1-stalw_0.0-cllw_0.0-2022-06-05_15:07:55

# sc
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/albert-base-v2/stable_cl_albert-per_bs_12-aug_1-gacc_1-stalw_0.0001-cllw_0.005-2022-05-18_19:21:39
#model_suffix=albert-base-v2/stable_cl_albert-per_bs_12-aug_1-gacc_1-stalw_0.0001-cllw_0.005-2022-05-18_19:21:39

# 1e-5
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/albert-base-v2/stable_cl_albert-per_bs_12-aug_1-gacc_1-stalw_0.001-cllw_0.001-2022-06-05_00:53:40
#model_suffix=albert-base-v2/stable_cl_albert-per_bs_12-aug_1-gacc_1-stalw_0.001-cllw_0.001-2022-06-05_00:53:40

## 3e-5
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/albert-base-v2/stable_cl_albert-per_bs_12-aug_1-gacc_1-stalw_0.005-cllw_0.0001-2022-06-05_00:52:32
#model_suffix=albert-base-v2/stable_cl_albert-per_bs_12-aug_1-gacc_1-stalw_0.005-cllw_0.0001-2022-06-05_00:52:32
#
## 3e-5
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/albert-base-v2/stable_cl_albert-per_bs_12-aug_1-gacc_1-stalw_0.001-cllw_0.001-2022-06-05_00:51:57
#model_suffix=albert-base-v2/stable_cl_albert-per_bs_12-aug_1-gacc_1-stalw_0.001-cllw_0.001-2022-06-05_00:51:57
#
## 1e-5
#model_name_or_path=/home/yuhai/workspace/qa/code/transformers/examples/pytorch/question-answering/results/squad_v2/albert-base-v2/stable_cl_albert-per_bs_12-aug_1-gacc_1-stalw_0.0001-cllw_0.005-2022-06-05_00:53:23
#model_suffix=albert-base-v2/stable_cl_albert-per_bs_12-aug_1-gacc_1-stalw_0.0001-cllw_0.005-2022-06-05_00:53:23

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
      --per_device_eval_batch_size 128 \
      --output_dir "$output_dir" \
      >> "$output_dir""/log.txt" 2>&1 \
      --cache_dir "$cache_dir"
done
