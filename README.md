# tPC-LM

Code and publicly available data for modeling sentence processing using temporal predictive coding.

**Create conda environment:** `conda env create -f tpc.yml`


To train model:
1. `cd src`
2. `sbatch train_batch.sh`

To extract_metrics:
1. `cd src`
2. `sbatch eval_batch_dundee.sh` for Dundee
3. `sbatch eval_batch_ns.sh` for Natural Stories

Analysis is performed for Dundee and Natural Stories corpora in `analysis/dundee.ipynb` and `analysis/natural_stories_analysis.ipynb` respectively
