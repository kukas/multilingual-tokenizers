#!/bin/bash

mkdir -p logs
exec > >(tee -a "logs/submit_finetune_model.log") 2>&1

mkdir -p ../job_outputs/model_finetuning_probe

# TODO: test -k d, does it do what I want?
qsub_command="qsub -e ../job_outputs/model_finetuning_probe -o ../job_outputs/model_finetuning_probe -A OPEN-26-22 -q qgpu -l select=1,walltime=6:00:00"

# for model in "multilingual_unigram_alpha0.0" "multilingual_unigram_alpha0.3" "Chung_12clusters"  "Chung_16clusters"  "Chung_20clusters"  "Chung_8clusters"
# do
#     for seed in 2001 2002 2003
#     do
#         echo "Submitting job for $model with seed $seed"
#         $qsub_command -N NLI-$model-$seed -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_nli.sh \
#             --model_name_or_path ../models/merged_1M_sizefix_3/$model \
#             --output_dir ../models/XNLI_probe/${model}_seed$seed/ \
#             --seed $seed --probe --precompute_model_outputs --keep_in_memory --use_custom_head --num_train_epochs 60 --save_steps 100 --eval_steps 100 --learning_rate 2e-3
#     done
# done

# for model in "multilingual_unigram_alpha0.3" "Chung_12clusters"  "Chung_16clusters"  "Chung_20clusters"  "Chung_8clusters" "multilingual_unigram_alpha0.0"
# do
#     for seed in 2002 2003 2004
#     do
#         echo "Submitting NER job for $model with seed $seed"

#         $qsub_command -N NER-$model-$seed -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_ner.sh \
#             --model_name_or_path ../models/merged_1M_sizefix_3/$model \
#             --output_dir ../models/NER_probe/${model}_seed$seed/ \
#             --seed $seed --probe --pad_to_max_length True --num_train_epochs 60 --save_steps 100 --eval_steps 100 --learning_rate 2e-3

#         echo "Submitting NER job for $model with seed $seed"
#         $qsub_command -N NER-$model-$seed-mean -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_ner.sh \
#             --model_name_or_path ../models/merged_1M_sizefix_3/$model \
#             --output_dir ../models/NER_mean_probe/${model}_seed$seed/ \
#             --seed $seed --probe --pad_to_max_length True --num_train_epochs 60 --save_steps 100 --eval_steps 100 --learning_rate 2e-3 --token_mean
#     done
# done


# for model in "multilingual_unigram_alpha0.3" "Chung_12clusters"  "Chung_16clusters"  "Chung_20clusters"  "Chung_8clusters" "multilingual_unigram_alpha0.0"
# do
#     for seed in 2001 2002 2003
#     do
#         echo "Submitting POS job for $model with seed $seed"
#         $qsub_command -N POS-$model-$seed -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_pos.sh \
#             --model_name_or_path ../models/merged_1M_sizefix_3/$model \
#             --output_dir ../models/POS_probe/${model}_seed$seed/ \
#             --seed $seed --probe --pad_to_max_length True --num_train_epochs 60 --save_steps 100 --eval_steps 100 --learning_rate 2e-3
#         $qsub_command -N POS-$model-$seed-mean -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_pos.sh \
#             --model_name_or_path ../models/merged_1M_sizefix_3/$model \
#             --output_dir ../models/POS_mean_probe/${model}_seed$seed/ \
#             --seed $seed --probe --pad_to_max_length True --num_train_epochs 60 --save_steps 100 --eval_steps 100 --learning_rate 2e-3 --token_mean
#     done
# done

# ----------------------
# run missing runs

# for model in "multilingual_unigram_alpha0.3" "Chung_12clusters"
# do
#     for seed in 2003
#     do
#         echo "Submitting NER job for $model with seed $seed"

#         $qsub_command -N NER-$model-$seed -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_ner.sh \
#             --model_name_or_path ../models/merged_1M_sizefix_3/$model \
#             --output_dir ../models/NER_probe/${model}_seed$seed/ \
#             --seed $seed --probe --pad_to_max_length True --num_train_epochs 60 --save_steps 100 --eval_steps 100 --learning_rate 2e-3

#     done
# done


for model in  "Chung_12clusters"
do
    for seed in 2004
    do
        echo "Submitting NER job for $model with seed $seed"


        $qsub_command -N NER-$model-$seed-mean -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_ner.sh \
            --model_name_or_path ../models/merged_1M_sizefix_3/$model \
            --output_dir ../models/NER_mean_probe/${model}_seed$seed/ \
            --seed $seed --probe --pad_to_max_length True --num_train_epochs 60 --save_steps 100 --eval_steps 100 --learning_rate 2e-3 --token_mean

    done
done


# for model in  "Chung_16clusters"
# do
#     for seed in 2002 2003
#     do
#         echo "Submitting NER job for $model with seed $seed"


#         $qsub_command -N NER-$model-$seed-mean -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_ner.sh \
#             --model_name_or_path ../models/merged_1M_sizefix_3/$model \
#             --output_dir ../models/NER_mean_probe/${model}_seed$seed/ \
#             --seed $seed --probe --pad_to_max_length True --num_train_epochs 60 --save_steps 100 --eval_steps 100 --learning_rate 2e-3 --token_mean

#     done
# done


# for model in "Chung_20clusters"
# do
#     for seed in 2002
#     do
#         echo "Submitting POS job for $model with seed $seed"
#         $qsub_command -N POS-$model-$seed -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_pos.sh \
#             --model_name_or_path ../models/merged_1M_sizefix_3/$model \
#             --output_dir ../models/POS_probe/${model}_seed$seed/ \
#             --seed $seed --probe --pad_to_max_length True --num_train_epochs 60 --save_steps 100 --eval_steps 100 --learning_rate 2e-3
#     done
# done


# for model in "Chung_16clusters"
# do
#     for seed in 2003
#     do
#         echo "Submitting POS job for $model with seed $seed"
#         $qsub_command -N POS-$model-$seed -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_pos.sh \
#             --model_name_or_path ../models/merged_1M_sizefix_3/$model \
#             --output_dir ../models/POS_probe/${model}_seed$seed/ \
#             --seed $seed --probe --pad_to_max_length True --num_train_epochs 60 --save_steps 100 --eval_steps 100 --learning_rate 2e-3
#     done
# done

# for model in "multilingual_unigram_alpha0.3"
# do
#     for seed in 2003
#     do
#         echo "Submitting POS job for $model with seed $seed"
#         $qsub_command -N POS-$model-$seed-mean -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_pos.sh \
#             --model_name_or_path ../models/merged_1M_sizefix_3/$model \
#             --output_dir ../models/POS_mean_probe/${model}_seed$seed/ \
#             --seed $seed --probe --pad_to_max_length True --num_train_epochs 60 --save_steps 100 --eval_steps 100 --learning_rate 2e-3 --token_mean
#     done
# done

# qsub_command="qsub -e ../job_outputs/model_finetuning_probe -o ../job_outputs/model_finetuning_probe -A OPEN-26-22 -q qgpu -l select=1,walltime=24:00:00"

# for model in "multilingual_unigram_alpha0.0" "multilingual_unigram_alpha0.3" "Chung_12clusters"  "Chung_16clusters"  "Chung_20clusters"  "Chung_8clusters"
# do
#     for seed in 2001 2002 2003
#     do
#         echo "Submitting job for $model with seed $seed"
#         $qsub_command -N NLI-$model-$seed -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_nli.sh \
#             --model_name_or_path ../models/merged_1M_sizefix_3/$model \
#             --output_dir ../models/XNLI_probe/${model}_seed$seed/ \
#             --seed $seed --probe --precompute_model_outputs --keep_in_memory --use_custom_head --num_train_epochs 60 --save_steps 100 --eval_steps 100 --learning_rate 2e-3
#     done
# done



# qsub_command="qsub -e ../job_outputs/model_finetuning_probe -o ../job_outputs/model_finetuning_probe -A OPEN-26-22 -q qgpu -l select=1,walltime=24:00:00"

# for model in "beta1.0" "beta0.9" "beta0.8" "beta0.7"
# do
#     for seed in 2001 2002 2003
#     do
#         echo "Submitting job for $model with seed $seed"
#         $qsub_command -N NLI-$model-$seed -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_nli.sh \
#             --model_name_or_path ../models/word_balancing/$model \
#             --output_dir ../models/XNLI_probe/${model}_seed$seed/ \
#             --seed $seed --probe --precompute_model_outputs --keep_in_memory --use_custom_head --num_train_epochs 60 --save_steps 100 --eval_steps 100 --learning_rate 2e-3
#     done
# done

# qsub_command="qsub -e ../job_outputs/model_finetuning -o ../job_outputs/model_finetuning -A OPEN-26-22 -q qgpu -l select=1,walltime=24:00:00"

# for model in "beta1.0" "beta0.9" "beta0.8" "beta0.7"
# do
#     for seed in 2001 2002 2003
#     do
#         echo "Submitting job for $model with seed $seed"
#         $qsub_command -N NLI-$model-$seed -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_nli.sh \
#             --model_name_or_path ../models/word_balancing/$model \
#             --output_dir ../models/XNLI_all/${model}_seed$seed/ \
#             --seed $seed
#     done
# done


# qsub_command="qsub -e ../job_outputs/model_finetuning_probe -o ../job_outputs/model_finetuning_probe -A OPEN-26-22 -q qgpu -l select=1,walltime=24:00:00"

# for model in "k_40" "k_80" "k_120"
# do
#     for seed in 2001 2002 2003
#     do
#         echo "Submitting job for $model with seed $seed"
#         $qsub_command -N NLI-$model-$seed -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_nli.sh \
#             --model_name_or_path ../models/document_clustering/$model \
#             --output_dir ../models/XNLI_probe/${model}_seed$seed/ \
#             --seed $seed --probe --precompute_model_outputs --keep_in_memory --use_custom_head --num_train_epochs 60 --save_steps 100 --eval_steps 100 --learning_rate 2e-3
#     done
# done

# qsub_command="qsub -e ../job_outputs/model_finetuning -o ../job_outputs/model_finetuning -A OPEN-26-22 -q qgpu -l select=1,walltime=24:00:00"

# for model in "k_40" "k_80" "k_120"
# do
#     for seed in 2001 2002 2003
#     do
#         echo "Submitting job for $model with seed $seed"
#         $qsub_command -N NLI-$model-$seed -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_nli.sh \
#             --model_name_or_path ../models/document_clustering/$model \
#             --output_dir ../models/XNLI_all/${model}_seed$seed/ \
#             --seed $seed
#     done
# done







# qsub_command="qsub -e ../job_outputs/model_finetuning -o ../job_outputs/model_finetuning -A OPEN-26-22 -q qgpu -l select=1,walltime=24:00:00"

# # for model in "beta1.0"
# # do
# #     for seed in 2002 #2003 2004
# #     do
# #         echo "Submitting NER job for $model with seed $seed"

# #         $qsub_command -N NER-$model-$seed -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_ner.sh \
# #             --model_name_or_path ../models/word_balancing/$model \
# #             --output_dir ../models/NER_all/${model}_seed$seed/ \
# #             --seed $seed

# #         echo "Submitting NER_probe job for $model with seed $seed"
# #         $qsub_command -N NER-$model-$seed-probe -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_ner.sh \
# #             --model_name_or_path ../models/word_balancing/$model \
# #             --output_dir ../models/NER_probe/${model}_seed$seed/ \
# #             --seed $seed --probe --pad_to_max_length True --num_train_epochs 60 --save_steps 100 --eval_steps 100 --learning_rate 2e-3

# #         echo "Submitting NER_mean_probe job for $model with seed $seed"
# #         $qsub_command -N NER-$model-$seed-mean -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_ner.sh \
# #             --model_name_or_path ../models/word_balancing/$model \
# #             --output_dir ../models/NER_mean_probe/${model}_seed$seed/ \
# #             --seed $seed --probe --pad_to_max_length True --num_train_epochs 60 --save_steps 100 --eval_steps 100 --learning_rate 2e-3 --token_mean
# #     done
# # done


# for model in "beta1.0"
# do
#     for seed in 2001 #2002 2003
#     do
#         echo "Submitting POS job for $model with seed $seed"
#         $qsub_command -N POS-$model-$seed -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_pos.sh \
#             --model_name_or_path ../models/word_balancing/$model \
#             --output_dir ../models/POS_all/${model}_seed$seed/ \
#             --seed $seed

#         # echo "Submitting POS_probe job for $model with seed $seed"
#         # $qsub_command -N POS-$model-$seed-probe -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_pos.sh \
#         #     --model_name_or_path ../models/word_balancing/$model \
#         #     --output_dir ../models/POS_probe/${model}_seed$seed/ \
#         #     --seed $seed --probe --pad_to_max_length True --num_train_epochs 60 --save_steps 100 --eval_steps 100 --learning_rate 2e-3

#         # echo "Submitting POS_mean_probe job for $model with seed $seed"
#         # $qsub_command -N POS-$model-$seed-mean -- /home/balharj/multilingual-tokenizers/scripts/finetune_model_pos.sh \
#         #     --model_name_or_path ../models/word_balancing/$model \
#         #     --output_dir ../models/POS_mean_probe/${model}_seed$seed/ \
#         #     --seed $seed --probe --pad_to_max_length True --num_train_epochs 60 --save_steps 100 --eval_steps 100 --learning_rate 2e-3 --token_mean
#     done
# done