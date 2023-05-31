#!/bin/bash
source ~/.bashrc
set -eux

export TRANSFORMERS_CACHE="/scratch/project/open-26-22/balharj/cache/huggingface"
export HF_DATASETS_CACHE="/scratch/project/open-26-22/balharj/cache/huggingface/datasets"
export HF_METRICS_CACHE="/scratch/project/open-26-22/balharj/cache/huggingface/metrics"
mkdir -p $TRANSFORMERS_CACHE

SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd $SCRIPTPATH/../src

# activate conda for subshell execution
CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate tokenizers

torchrun --nproc_per_node 2 --nnodes 1 --rdzv_backend c10d --rdzv_endpoint localhost:0 run_mlm.py \
    --do_train --do_eval \
    --seed 42 \
    --model_type xlm-roberta \
    --max_seq_length 128 \
    --line_by_line \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 32 \
    --learning_rate 5e-4 \
    --save_steps 200 \
    --eval_steps 200 \
    --fp16 \
    --torch_compile True \
    --ddp_find_unused_parameters False \
    --max_steps 10000 \
    --warmup_steps 500 \
    --load_best_model_at_end \
    --eval_accumulation_steps 1 \
    --evaluation_strategy steps \
    --logging_steps 10 \
    --save_total_limit 5 \
    --per_device_eval_batch_size 64 \
    --dataloader_num_workers 14 \
    --preprocessing_num_workers 14 \
    --config_overrides "vocab_size=120002,hidden_size=768,num_hidden_layers=8,num_attention_heads=6,max_position_embeddings=514" \
    "$@" > ../scripts/logs/pretrain_model_$(date +'%Y%m%d-%H%M%S').log 2>&1
