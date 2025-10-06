#!/bin/bash

#SBATCH -t 06:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1

python3 extract_metrics.py --K 0 --model_path "../results/trained_models_linear_0_512_new/epoch_199.pt" --batch_size 64 --corpus_path "./corpora/dundee.csv" --savedir "../results/dundee_evals_linear_0_512_new" 
