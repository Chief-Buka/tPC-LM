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

Repo built with reference to [temporal predictive coding](https://github.com/C16Mftang/temporal-predictive-coding/tree/main).

If you use any part of the implementation, please cite us at:
```
@article{OHAMS2026104705,
	abstract = {Computational approaches to prediction in online sentence processing tend to be dominated by computation-level surprisal theory, offering few insights into underlying cognitive mechanisms. Conversely, predictive coding is an algorithmic theory grounded in neuroscience, but it has rarely been employed in the study of language processing, in part because its areas of application have not involved sequential processing. Building on a recently proposed temporal predictive coding model, we present what is to our knowledge the first exploration of sequential predictive coding in broad-coverage online sentence processing. We investigate our model at non-toy scale using naturally occurring language, establishing its cognitive validity via comparison with reading times, and we link measurable aspects of the model to cognitive discussions of mechanism for prediction in language processing. Our results suggest that sequential predictive coding models are a valuable complement to surprisal theory as a route to progress on process-oriented theories of language comprehension.},
	author = {Chiebuka Ohams and Sathvik Nair and Shohini Bhattasali and Philip Resnik},
	doi = {https://doi.org/10.1016/j.jml.2025.104705},
	issn = {0749-596X},
	journal = {Journal of Memory and Language},
	keywords = {Predictive coding, Incremental sentence processing, Prediction in language processing},
	pages = {104705},
	title = {A predictive coding model for online sentence processing},
	url = {https://www.sciencedirect.com/science/article/pii/S0749596X25000981},
	volume = {146},
	year = {2026},
	bdsk-url-1 = {https://www.sciencedirect.com/science/article/pii/S0749596X25000981},
	bdsk-url-2 = {https://doi.org/10.1016/j.jml.2025.104705}}
```
