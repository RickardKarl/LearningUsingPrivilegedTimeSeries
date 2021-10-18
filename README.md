
# PAPER TITLE



This repository is the official implementation of [title](). 

### To-do list
- [ ] Clean up code for distll-experiments?
- [ ] Add arXiv link

![](images/illustration-crop.png)

### Abstract
- [ ] Add abstract

## Requirements

Required libraries found in *requirements.txt*

## Models
Baseline and LUPTS are implemented using sklearn, the code is found in */src/model/*

## Evaluation

### Synthethic

To re-produce experiments, run */notebooks/synthetic.ipynb*
Necessary experiment code is found in */src/synthetic/*

### Forecasting Air Quality

To re-produce experiments, run */notebooks/fivecities.ipynb*
Necessary experiment code is found in */src/fivecities/*

The data is found in */data/fivecities/*, but can also be downloaded from [here](https://archive.ics.uci.edu/ml/datasets/PM2.5+Data+of+Five+Chinese+Cities).

### Modeling Progression of Chronic Disease

**Note**: For the Alzheimer’s and Multiple myeloma progression modeling tasks, the data is not publicly available, but the code which produced the results is still found in this repository. 

#### Alzheimer's progression modelling
Code is found in */notebooks/ADNI.ipynb* and */src/adni/*

#### Multiple myeloma progression modelling
Code is found in */notebooks/mm-prfs.ipynb* and */notebooks/mm-tr.ipynb*

## Bibtex entry for citation

- [ ] To be filled out
