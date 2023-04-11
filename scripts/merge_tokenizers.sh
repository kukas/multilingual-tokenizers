#!/bin/bash
#SBATCH --mem=64g
#SBATCH --cpus-per-task=32
#SBATCH --time=144:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=balhar.j@gmail.com
#SBATCH --output=/home/balhar/my-luster/contextual-tokenizers/job_outputs/merge_tokenizers/merging_%j.out

set -eux

cd /home/balhar/my-luster/contextual-tokenizers/src
# TODO: create a virtual environment for this
# source ../../eis/bin/activate
python -u merge_tokenizers.py "$@"