#!/bin/bash

mkdir -p logs
exec > >(tee -a "logs/submit_finetune_model.log") 2>&1

mkdir -p ../job_outputs/model_training

# qsub_command="qsub -A OPEN-26-22 -q qgpu -l select=1,walltime=4:00:00"

# for model in "multilingual_unigram_alpha0.0" "multilingual_unigram_alpha0.3" "Chung_12clusters"  "Chung_16clusters"  "Chung_20clusters"  "Chung_8clusters"
# do
#     for seed in 2001 2002 2003
#     do
#         echo "Submitting job for $model with seed $seed"
#         $qsub_command -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_nli.sh \
#             --model_name_or_path ../models/merged_1M_sizefix_3/$model \
#             --output_dir ../models/XNLI_all/${model}_seed$seed/ \
#             --seed $seed
#     done
# done


# qsub_command="qsub -A OPEN-26-22 -q qgpu -l select=1,walltime=3:00:00"

# for model in "multilingual_unigram_alpha0.3" "Chung_12clusters"  "Chung_16clusters"  "Chung_20clusters"  "Chung_8clusters" "multilingual_unigram_alpha0.0"
# do
#     for seed in 2001 2002 2003
#     do
#         echo "Submitting NER job for $model with seed $seed"
#         $qsub_command -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_ner.sh \
#             --model_name_or_path ../models/merged_1M_sizefix_3/$model \
#             --output_dir ../models/NER_all/${model}_seed$seed/ \
#             --seed $seed
#     done
# done

# qsub_command="qsub -A OPEN-26-22 -q qgpu -l select=1,walltime=3:00:00"

# for model in "multilingual_unigram_alpha0.3" "Chung_12clusters"  "Chung_16clusters"  "Chung_20clusters"  "Chung_8clusters" "multilingual_unigram_alpha0.0"
# do
#     for seed in 2001 2002 2003
#     do
#         echo "Submitting POS job for $model with seed $seed"
#         $qsub_command -N POS-$model-$seed -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_pos.sh \
#             --model_name_or_path ../models/merged_1M_sizefix_3/$model \
#             --output_dir ../models/POS_all/${model}_seed$seed/ \
#             --seed $seed
#     done
# done

# run jobs that failed
# qsub_command="qsub -A OPEN-26-22 -q qgpu -l select=1,walltime=4:00:00"
# model=Chung_20clusters
# seed=2001

# $qsub_command -N NER-$model-$seed -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_ner.sh \
#     --model_name_or_path ../models/merged_1M_sizefix_3/$model \
#     --output_dir ../models/NER_all/${model}_seed$seed/ \
#     --seed $seed


