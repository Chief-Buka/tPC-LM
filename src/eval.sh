#!/bin/bash

python3 extract_metrics.py --model_path "../results/trained_models/epoch_19.pt" --corpus_path "corpora/brown/test_brown.csv" --savedir ../results/evals
