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

Analysis is performed for Dundee and Natural Stories corpora in `analysis/dundee.ipynb` and `analysis/natural_stories_analysis.ipynb` respectively.

RTs for the Dundee corpus are averaged gaze durations provided by [Wilcox et al. (Cog Sci, 2020)](https://github.com/wilcoxeg/neural-networks-read-times) from their Google Drive link. RTs from the Natural Stories corpus are provided by [Futrell et al. (Language Resources & Evaluation, 2021)](https://github.com/languageMIT/naturalstories).

If you use any part of the implementation, please cite us at:
```
@article{ohams2025predictive,
  title={A Predictive Coding Model for Online Sentence Processing},
  author={Ohams, Chiebuka and Nair, Sathvik and Bhattasali, Shohini and Resnik, Philip},
  year={2026},
  publisher={Journal of Memory and Language}
}
```
