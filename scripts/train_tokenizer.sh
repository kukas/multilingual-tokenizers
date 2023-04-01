#!/bin/bash
#SBATCH --mem=64g
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=balhar.j@gmail.com
#SBATCH --output=/home/balhar/my-luster/contextual-tokenizers/job_outputs/tokenizer_training/training_%j.out

set -eux

cd /home/balhar/my-luster/contextual-tokenizers/src
# TODO: create a virtual environment for this
# source ../../eis/bin/activate

python train_tokenizer.py "$@"