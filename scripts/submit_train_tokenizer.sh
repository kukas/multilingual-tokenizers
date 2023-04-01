#!/bin/bash
set -eux

mkdir -p /home/$USER/my-luster/contextual-tokenizers/job_outputs/tokenizer_training

data_dir="/home/$USER/my-luster/contextual-tokenizers/data/cc100_subsample0.062_alpha0.3"
langs=("en" "es" "el" "zh-Hans" "tr" "ar" "sw" "hi" "mr" "ur" "ta" "te" "th" "ru" "bg" "he" "ka" "vi" "fr" "de")

output_dir="/home/$USER/my-luster/contextual-tokenizers/tokenizers/sweep_1/"
for lang in "${langs[@]}"; do
    mem="244g"
    case $lang in
        zh-Hans|ka|mr|ar|tr|ur|te|sw|el|bg) mem="124g" ;;
    esac
    for vocab_size in 500 1000 2000 4000 8000 16000 32000 64000; do
        for model in unigram; do
            sbatch --cpus-per-task=8 --mem=$mem train_tokenizer.sh --num_threads 8 --input $data_dir/$lang.txt.train --output_dir $output_dir --output_prefix $lang --vocab_size $vocab_size --model_type $model --character_coverage 0.9995
        done
    done
    # sbatch $sbatch_args train_tokenizer.py --input $data_dir/$lang.txt --output $data_dir/tokenizer.pkl --vocab_size 10000
    # sbatch --cpus-per-task=64 --mem=244g scripts/train_tokenizer.sh --input {inputs} --output_dir /home/balhar/my-luster/contextual-tokenizers/tokenizers/{dataset}/ --output_prefix {prefix} --vocab_size {vocab_size} --model_type bpe --character_coverage 1.0 --max_num_sentences {max_num_sentences} --huggingface"
done
# lang="ta"
# sbatch train_tokenizer.sh --input $data_dir/$lang.txt --output_dir $output_dir --output_prefix $lang --vocab_size 8000 --model_type bpe --character_coverage 0.9995 --overwrite

# if lang one of zh-Hans	ka|mr|ar|tr|ur|te|sw
