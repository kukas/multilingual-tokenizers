#!/bin/bash
#SBATCH --mem=64g
#SBATCH --cpus-per-task=16
#SBATCH --time=13-23
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --constraint="gpuram40G|gpuram48G"
#SBATCH --gres=gpu:3
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=balhar.j@gmail.com
#SBATCH --output=/home/balhar/my-luster/contextual-tokenizers/job_outputs/model_training/training_%j.out

set -eux

export TRANSFORMERS_CACHE="/lnet/troja/work/people/balhar/cache/huggingface"
export HF_DATASETS_CACHE="/lnet/troja/work/people/balhar/cache/huggingface/datasets"
export HF_METRICS_CACHE="/lnet/troja/work/people/balhar/cache/huggingface/metrics"
mkdir -p $TRANSFORMERS_CACHE

cd /home/balhar/my-luster/contextual-tokenizers/src

python run_mlm.py \
    --do_train --do_eval \
    --seed 42 \
    --model_type xlm-roberta \
    --max_seq_length 128 \
    --line_by_line \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-4 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --fp16 \
    --max_steps 15000 \
    --warmup_steps 500 \
    --load_best_model_at_end \
    --warmup_ratio 0.01 \
    --eval_accumulation_steps 1 \
    --evaluation_strategy steps \
    --logging_steps 10 \
    --save_total_limit 5 \
    --per_device_eval_batch_size 64 \
    --dataloader_num_workers 14 \
    --preprocessing_num_workers 14 \
    --config_overrides "vocab_size=120002,hidden_size=512,num_hidden_layers=6,num_attention_heads=4,max_position_embeddings=130" \
    "$@"
