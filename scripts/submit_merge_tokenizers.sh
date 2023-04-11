#!/bin/bash
mkdir -p logs
exec > >(tee -a "logs/submit_merge_tokenizers.log") 2>&1

mkdir -p ../job_outputs/merge_tokenizers

common_args="--input ../data/cc100_1M/* --output_prefix merged --sample_lines 10000 --target_vocab_size 120000"
for k in 20 40 80 160 320; do
    echo "k = $k"
    sbatch merge_tokenizers.sh $common_args --group_regex "cluster_\d+" --tokenizers ../tokenizers/cc100_clustered/cc100_1M_k\=$k/cluster_* --output_dir ../tokenizers/cc100_clustered/cc100_1M_k\=$k
done