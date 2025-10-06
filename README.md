# tPC-LM

Code and publicly available data for modeling sentence processing using temporal predictive coding.

**Create conda environment:** conda env create -f tpc.yml


To train model:
1. `cd src`
2. `sbatch train_batch.sh`

To extract_metrics:
1. `cd src`
2. `sbatch eval_batch.sh`

Analysis is performed for Natural Stories and Dundee corpora in `analysis/natural_stories_analysis.ipynb` and `analysis/dundee.ipynb` respectively
