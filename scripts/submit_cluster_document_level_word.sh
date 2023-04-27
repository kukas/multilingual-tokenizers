#!/bin/bash

mkdir -p logs
exec > >(tee -a "logs/submit_cluster_document_level_word.log") 2>&1

mkdir -p ../data/cc100_subsample0.062_alpha0.3_en_clustered/

# common_args="--input /home/balhar/my-luster/contextual-tokenizers/data/cc100_subsample0.062_alpha0.3/en.txt.train --max_documents 40000 --output_dir ../data/cc100_subsample0.062_alpha0.3_en_clustered/ --output_prefix cluster_sklearn"
# common_args="$common_args --vectorizer word --max_df 0.5 --svd_components 300"
# sbatch --mem 64g --cpus-per-task=8 cluster_document_level.sh $common_args -k 16


common_args="--input /home/balhar/my-luster/contextual-tokenizers/data/cc100_subsample0.062_alpha0.3/en.txt.train --output_dir ../data/cc100_subsample0.062_alpha0.3_en_clustered/ --output_prefix cluster_sklearn"
common_args="$common_args --vectorizer word --max_df 0.5 --svd_components 300"
sbatch --mem 244g --cpus-per-task=8 cluster_document_level.sh $common_args -k 16


# common_args="--input /home/balhar/my-luster/contextual-tokenizers/data/cc100_subsample0.062_alpha0.3/en.txt.train --max_documents 40000 --output_dir ../data/cc100_subsample0.062_alpha0.3_en_clustered/ --output_prefix cluster_sklearn"
# common_args="$common_args --vectorizer char --ngram_range 3 10 --max_df 0.5"
# sbatch --mem 64g --cpus-per-task=8 cluster_document_level.sh $common_args -k 16

# common_args="--input /home/balhar/my-luster/contextual-tokenizers/data/short_cc100/cs.txt --max_documents 40000 --output_dir ../data/cc100_cs_clustered/ --output_prefix cluster_sklearn"
# common_args="$common_args --vectorizer word --max_df 0.5 --svd_components 300"
# sbatch --mem 64g --cpus-per-task=8 cluster_document_level.sh $common_args -k 16


# common_args="--input /home/balhar/my-luster/contextual-tokenizers/data/cc100_subsample0.062_alpha0.3/en.txt.train /home/balhar/my-luster/contextual-tokenizers/data/short_cc100/cs.txt --max_documents 30000 --output_dir ../data/cc100_en_cs_clustered/ --output_prefix cluster_sklearn"
# common_args="$common_args --vectorizer word --max_df 0.5"
# sbatch --mem 64g --cpus-per-task=8 cluster_document_level.sh $common_args -k 16


# common_args="--input /home/balhar/my-luster/contextual-tokenizers/data/cc100_subsample0.062_alpha0.3/en.txt.train /home/balhar/my-luster/contextual-tokenizers/data/short_cc100/cs.txt /home/balhar/my-luster/contextual-tokenizers/data/short_cc100/sk.txt --max_documents 30000 --output_dir ../data/cc100_en_cs_sk_clustered/ --output_prefix cluster_sklearn"
# common_args="$common_args --vectorizer word --max_df 0.10 --svd_components 300"
# sbatch --mem 64g --cpus-per-task=8 cluster_document_level.sh $common_args -k 16

# common_args="--input /home/balhar/my-luster/contextual-tokenizers/data/cc100_1M/*.txt.train --max_documents 30000 --output_dir ../data/cc100_1M_clustered/ --output_prefix cluster_sklearn"
# common_args="$common_args --vectorizer word --max_df 0.01 --svd_components 500"
# sbatch --mem 244g --cpus-per-task=16 cluster_document_level.sh $common_args -k 16


# common_args="--input /home/balhar/my-luster/contextual-tokenizers/data/cc100_subsample0.062_alpha0.3/en.txt.train /home/balhar/my-luster/contextual-tokenizers/data/short_cc100/cs.txt /home/balhar/my-luster/contextual-tokenizers/data/short_cc100/sk.txt --max_documents 30000 --output_dir ../data/cc100_en_cs_sk_clustered/ --output_prefix cluster_sklearn"
# common_args="$common_args --vectorizer char --ngram_range 3 10 --max_df 0.5 --svd_components 300"
# sbatch --mem 244g --cpus-per-task=8 cluster_document_level.sh $common_args -k 16


# common_args="--input /home/balhar/my-luster/contextual-tokenizers/data/cc100_1M/*.txt.train --output_dir ../data/cc100_1M_clustered/ --output_prefix cluster_sklearn"
# common_args="$common_args --vectorizer word --max_df 0.01 --svd_components 500"
# sbatch --mem 244g --cpus-per-task=16 cluster_document_level.sh $common_args -k 16
