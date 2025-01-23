#!/bin/bash

#SBATCH -t 1-00:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1


python3 train.py --K 0 --y_size 300 --f_type linear --g_type linear --x_size 512 --batch_size 64  --inf_iters 50 --delta_t_x 1e-1 --delta_t_w 3e-4 --device cuda --train_data_path "./corpora/train_brown.csv" --val_data_path "./corpora/val_brown.csv" --epochs 200 --savedir ../results/trained_models_linear_0_512_new
