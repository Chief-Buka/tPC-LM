#!/bin/bash

python3 train.py --autoregressive --y_size 300 --x_size 600 --batch_size 64 --inf_iters 150 --delta_t_x 1e-1 --delta_t_w 5e-4 --device cuda --train_data_path "corpora/brown/train_brown.csv" --val_data_path "corpora/brown/val_brown.csv" --epochs 100 --savedir ../results/trained_models
