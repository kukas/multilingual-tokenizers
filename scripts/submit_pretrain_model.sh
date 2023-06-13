#!/bin/bash

mkdir -p logs
exec > >(tee -a "logs/submit_pretrain_model.log") 2>&1

mkdir -p ../job_outputs/model_training
mkdir -p ../models

# qsub_command="qsub -A OPEN-26-22 -q qgpu_exp -l select=2,walltime=1:00:00"
qsub_command="qsub -m bea -M balhar.j@gmail.com -A OPEN-26-22 -q qgpu -l select=2,walltime=48:00:00"

# for clusters in 8 12 16 20; do
#     $qsub_command -- /home/balharj/multilingual-tokenizers/scripts/pretrain_model.sh \
#         --output_dir ../models/merged_1M_sizefix_3/Chung_${clusters}clusters/ \
#         --train_file ../data/cc100_subsample0.1_alpha0.3/*.train.txt \
#         --validation_file ../data/cc100_subsample0.1_alpha0.3/*.valid.txt \
#         --max_eval_samples 20000 \
#         --sentencepiece_path ../tokenizers/merged_1M_sizefix/Chung_${clusters}clusters/m.model
# done

# $qsub_command -- /home/balharj/multilingual-tokenizers/scripts/pretrain_model.sh \
#         --output_dir ../models/merged_1M_sizefix_3/multilingual_unigram_alpha0.3/ \
#         --train_file ../data/cc100_subsample0.1_alpha0.3/*.train.txt \
#         --validation_file ../data/cc100_subsample0.1_alpha0.3/*.valid.txt \
#         --max_eval_samples 20000 \
#         --sentencepiece_path ../tokenizers/cc100_subsample0.1_alpha0.3/en-es-el-zh-Hans-tr-ar-sw-hi-mr-ur-ta-te-th-ru-bg-he-ka-vi-fr-de_unigram_120000_coverage0.9995/m.model

# $qsub_command -- /home/balharj/multilingual-tokenizers/scripts/pretrain_model.sh \
#         --output_dir ../models/merged_1M_sizefix_3/multilingual_unigram_alpha0.0/ \
#         --train_file ../data/cc100_subsample0.1_alpha0.3/*.train.txt \
#         --validation_file ../data/cc100_subsample0.1_alpha0.3/*.valid.txt \
#         --max_eval_samples 20000 \
#         --sentencepiece_path ../tokenizers/clustered_1M/en-es-el-zh-Hans-tr-ar-sw-hi-mr-ur-ta-te-th-ru-bg-he-ka-vi-fr-de_unigram_120000vocab_0.9995coverage/m.model

# $qsub_command -- /home/balharj/multilingual-tokenizers/scripts/pretrain_model.sh \
#         --output_dir ../models/merged_1M_sizefix_3/multilingual_bpe_alpha0.0/ \
#         --train_file ../data/cc100_subsample0.1_alpha0.3/*.train.txt \
#         --validation_file ../data/cc100_subsample0.1_alpha0.3/*.valid.txt \
#         --max_eval_samples 20000 \
#         --sentencepiece_path ../tokenizers/clustered_1M/en-es-el-zh-Hans-tr-ar-sw-hi-mr-ur-ta-te-th-ru-bg-he-ka-vi-fr-de_unigram_120000vocab_0.9995coverage/m.model


# $qsub_command -- /home/balharj/multilingual-tokenizers/scripts/pretrain_model.sh \
#         --output_dir ../models/merged_1M_sizefix_3/multilingual_limi_unigram/ \
#         --train_file ../data/cc100_subsample0.1_alpha0.3/*.train.txt \
#         --validation_file ../data/cc100_subsample0.1_alpha0.3/*.valid.txt \
#         --max_eval_samples 20000 \
#         --tokenizer_name ../tokenizers/limi/sp-unigram/ar-tr-zh-el-es-en-sw-hi-mr-ur-ta-te-th-ru-bg-he-ka-vi-fr-de/alpha-0.25_N-120000


# $qsub_command -- /home/balharj/multilingual-tokenizers/scripts/pretrain_model.sh \
#         --output_dir ../models/merged_1M_sizefix_3/multilingual_limi_unigram-merged/ \
#         --train_file ../data/cc100_subsample0.1_alpha0.3/*.train.txt \
#         --validation_file ../data/cc100_subsample0.1_alpha0.3/*.valid.txt \
#         --max_eval_samples 20000 \
#         --tokenizer_name ../tokenizers/limi/sp-unigram-merged/ar-tr-zh-el-es-en-sw-hi-mr-ur-ta-te-th-ru-bg-he-ka-vi-fr-de/alpha-0.25_N-120000

# $qsub_command -- /home/balharj/multilingual-tokenizers/scripts/pretrain_model.sh \
#         --output_dir ../models/merged_1M_sizefix_3/multilingual_limi_bpe/ \
#         --train_file ../data/cc100_subsample0.1_alpha0.3/*.train.txt \
#         --validation_file ../data/cc100_subsample0.1_alpha0.3/*.valid.txt \
#         --max_eval_samples 20000 \
#         --tokenizer_name ../tokenizers/limi/sp-bpe/ar-tr-zh-el-es-en-sw-hi-mr-ur-ta-te-th-ru-bg-he-ka-vi-fr-de/alpha-0.25_N-120000

# word balancing
# for beta in 1.0 0.9 0.8 0.7; do
#         $qsub_command -- /home/balharj/multilingual-tokenizers/scripts/pretrain_model.sh \
#                 --output_dir ../models/word_balancing/beta$beta \
#                 --train_file ../data/cc100_subsample0.1_alpha0.3/*.train.txt \
#                 --validation_file ../data/cc100_subsample0.1_alpha0.3/*.valid.txt \
#                 --max_eval_samples 20000 \
#                 --sentencepiece_path ../tokenizers/cc100_1M_words_resampled/multilingual_beta${beta}_unigram_120000vocab_0.9995coverage/m.model
# done

# document clustering
for k in 40 80 120; do
        $qsub_command -- /home/balharj/multilingual-tokenizers/scripts/pretrain_model.sh \
                --output_dir ../models/document_clustering/k_${k} \
                --train_file ../data/cc100_subsample0.1_alpha0.3/*.train.txt \
                --validation_file ../data/cc100_subsample0.1_alpha0.3/*.valid.txt \
                --max_eval_samples 20000 \
                --sentencepiece_path "../tokenizers/cc100_subsample0.062_alpha0.3_word_clustered/clusters_inference_vectorizer=word_max_documents=30000_max_df=0.01_svd_components=500_max_documents=500000/k_${k}_maximize_cpt/merged_target_vocab_size=120000_sample_lines=10000/m.model"
done