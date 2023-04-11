#!/bin/bash

# Cluster the documents using the document-level clustering method

# common_args="--input ../data/cc100_1M/*.txt.train --tokenizer ../tokenizers/clustered_bpe_1M/merged_bpe_504959vocab_0.9995coverage/m.model --output_dir ../data/cc100_clustered/ --output_prefix cc100_1M"
# sbatch cluster_document_level.sh $common_args -k 10
# sbatch cluster_document_level.sh $common_args -k 20
# sbatch cluster_document_level.sh $common_args -k 40
# sbatch cluster_document_level.sh $common_args -k 80
# sbatch cluster_document_level.sh $common_args -k 160
# sbatch cluster_document_level.sh $common_args -k 320

# Train new tokenizers on the clustered documents
mkdir -p ../job_outputs/data_cluster_tokenizer_training_2

model="unigram"
for k in 20 40 80 160 320; do
    echo "k = $k"
    for i in $(seq 0 $(($k - 1))); do
        input="../data/cc100_clustered/cc100_1M_k=$k/cluster_$i.txt"
        num_lines=$(wc -l < $input)
        echo "cluster $i has $num_lines lines"
        mem=$(($num_lines / 1000000 * 10 + 15))
        cpus=$(($num_lines / 1000000 + 1))
        echo "mem = $mem, cpus = $cpus"
        prefix="cluster_$i"
        for vocab_size in $(seq 1000 1000 20000); do
            output_dir="../tokenizers/cc100_clustered/cc100_1M_k=$k/"
            # check if the tokenizer already exists
            model_path="$output_dir/cluster_${i}_${model}_${vocab_size}vocab_0.9995coverage/m.model"
            if [ -f $model_path ]; then
                echo "Skipping $model_path"
                continue
            fi
            sbatch --output=../job_outputs/data_cluster_tokenizer_training_2/training_%j.out --cpus-per-task=$cpus --mem=${mem}g train_tokenizer.sh --num_threads $cpus --input $input --output_dir $output_dir --output_prefix $prefix --vocab_size $vocab_size --model_type $model --character_coverage 0.9995
        done
    done
    # show the number of lines for each cluster
    # sbatch train_tokenizer.sh -i ../data/cc100_clustered/cc100_1M_${k}clusters.txt.train -o ../tokenizers/clustered_bpe_1M/merged_bpe_504959vocab_0.9995coverage/m.model -k $k
done