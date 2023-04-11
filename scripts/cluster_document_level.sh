#!/bin/bash
#SBATCH --mem=244g
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=balhar.j@gmail.com
#SBATCH --output=/home/balhar/my-luster/contextual-tokenizers/job_outputs/data_clustering/clustering_%j.out

set -eux

cd /home/balhar/my-luster/contextual-tokenizers/src
# TODO: create a virtual environment for this
# source ../../eis/bin/activate
python -u cluster_document_level_sklearn.py "$@"