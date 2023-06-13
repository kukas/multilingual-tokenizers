#!/bin/bash

# activate conda for subshell execution
# check if conda is already activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    source ~/.bashrc
    CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
    source $CONDA_PATH/etc/profile.d/conda.sh
    conda activate tokenizers
fi

export TRANSFORMERS_CACHE="/scratch/project/open-26-22/balharj/cache/huggingface"
export HF_DATASETS_CACHE="/scratch/project/open-26-22/balharj/cache/huggingface/datasets"
export HF_METRICS_CACHE="/scratch/project/open-26-22/balharj/cache/huggingface/metrics"
mkdir -p $TRANSFORMERS_CACHE

SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd $SCRIPTPATH/../src


# set wandb environment variables
export WANDB_PROJECT="multilingual-tokenizers-finetune-ner"
export WANDB_LOG_MODEL="false"
export WANDB_WATCH="false"

# if the first argument is --distributed, then run with torchrun and remove the argument from the list
if [ "$1" = "--distributed" ]; then
    # num_gpus=$(($(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c)+1)) # counts the number of commas and adds 1
    num_gpus=$(nvidia-smi  -L | wc -l) # better way
    run_command="torchrun --nproc_per_node $num_gpus --nnodes 1 --rdzv_backend c10d --rdzv_endpoint localhost:0"
    shift
else
    run_command="python -u"
fi

# next argument must be model_name_or_path
if [ "$1" = "--model_name_or_path" ]; then
    model_name_or_path="$2"
    shift
    shift
else
    echo "ERROR: model_name_or_path must be specified"
    exit 1
fi

# next argument must be output_dir
if [ "$1" = "--output_dir" ]; then
    output_dir="$2"
    shift
    shift
else
    echo "ERROR: output_dir must be specified"
    exit 1
fi

TOKENIZERS_PARALLELISM=false

echo "task NER"
echo "run_command $run_command"
echo "rest of args $@"
langs=("en" "ar" "el" "es" "tr" "zh" "sw" "hi" "mr" "ur" "ta" "te" "th" "ru" "bg" "he" "ka" "vi" "fr" "de")

for lang_src in ${langs[@]}
do
    output_dir_lang=$output_dir/$lang_src
    
    if [ ! -f "$output_dir_lang/pytorch_model.bin" ]; then
        $run_command run_ner.py \
            --do_train \
            --model_name_or_path $model_name_or_path \
            --output_dir $output_dir_lang \
            --dataset_name xtreme \
            --dataset_config_name PAN-X.$lang_src \
            --per_device_train_batch_size 512 \
            --per_device_eval_batch_size 512 \
            --learning_rate 5e-5 \
            --num_train_epochs 5 \
            --max_seq_length 128 \
            --load_best_model_at_end \
            --metric_for_best_model f1 \
            --save_steps 60 \
            --eval_steps 60 \
            --save_total_limit 1 \
            --evaluation_strategy steps \
            --eval_accumulation_steps 1 \
            --fp16 \
            --dataloader_num_workers 16 \
            "$@"
    fi
    echo "######################"
    echo "##### EVALUATION #####"
    echo "######################"

    for lang_tgt in ${langs[@]}
    do
        # now we use the finetuned model as the model_name_or_path 
        # output_dir is now the language-specific directory
        if [ ! -f "$output_dir_lang/$lang_tgt/eval_results.json" ]; then
            $run_command run_ner.py \
                --do_eval \
                --model_name_or_path $output_dir_lang \
                --output_dir $output_dir_lang/$lang_tgt \
                --dataset_name xtreme \
                --dataset_config_name PAN-X.$lang_tgt \
                --per_device_eval_batch_size 512 \
                --max_seq_length 128 \
                --eval_accumulation_steps 1 \
                --fp16 \
                --dataloader_num_workers 16 \
                --report_to "none" \
                "$@"
        fi
    done
done