#!/bin/bash

mkdir -p logs
exec > >(tee -a "logs/submit_pretrain_model.log") 2>&1

mkdir -p ../job_outputs/model_training
mkdir -p ../models

sbatch pretrain_model.sh \
    --output_dir ../models/merged_1M_sizefix_2/Chung_16clusters/ \
    --train_file ../data/cc100_subsample0.1_alpha0.3/*.txt \
    --validation_file ../data/cc100_subsample0.062_alpha0.3/*.valid \
    --max_train_samples 100_000 \
    --max_eval_samples 20000 \
    --sentencepiece_path ../tokenizers/merged_1M_sizefix/Chung_16clusters/m.model


# sbatch pretrain_model.sh \
#     --output_dir ../models/merged_1M_sizefix/Chung_8clusters/ \
#     --train_file ../data/cc100_subsample0.062_alpha0.3/*.train \
#     --validation_file ../data/cc100_subsample0.062_alpha0.3/*.valid \
#     --max_eval_samples 20000 \
#     --sentencepiece_path ../tokenizers/merged_1M_sizefix/Chung_8clusters/m.model


# sbatch pretrain_model.sh \
#     --output_dir ../models/merged_1M_sizefix/Chung_20clusters/ \
#     --train_file ../data/cc100_subsample0.062_alpha0.3/*.train \
#     --validation_file ../data/cc100_subsample0.062_alpha0.3/*.valid \
#     --max_eval_samples 20000 \
#     --sentencepiece_path ../tokenizers/merged_1M_sizefix/Chung_20clusters/m.model

# sbatch pretrain_model.sh \
#     --output_dir ../models/merged_1M_sizefix/Chung_12clusters/ \
#     --train_file ../data/cc100_subsample0.062_alpha0.3/*.train \
#     --validation_file ../data/cc100_subsample0.062_alpha0.3/*.valid \
#     --max_eval_samples 20000 \
#     --sentencepiece_path ../tokenizers/merged_1M_sizefix/Chung_12clusters/m.model

# sbatch pretrain_model.sh \
#     --output_dir ../models/cc100_1M/unigram_alpha0.0/ \
#     --train_file ../data/cc100_subsample0.062_alpha0.3/*.train \
#     --validation_file ../data/cc100_subsample0.062_alpha0.3/*.valid \
#     --max_eval_samples 20000 \
#     --sentencepiece_path ../tokenizers/clustered_1M/en-es-el-zh-Hans-tr-ar-sw-hi-mr-ur-ta-te-th-ru-bg-he-ka-vi-fr-de_unigram_120000vocab_0.9995coverage/m.model
