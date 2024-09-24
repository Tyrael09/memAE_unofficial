#!/bin/bash
#datatype=${1?Error: which dataset am I using?}
#datapath=${2?Error: where is the dataset}
#expdir=${3?Error: where to save the experiment}

python Train.py --dataset_path /local/scratch/hendrik/memAE/ --dataset_type Cataract --version 0 --EntropyLossWeight 0 --lr 1e-4 --exp_dir /local/scratch/hendrik/memAE/Avenue/checkpoints