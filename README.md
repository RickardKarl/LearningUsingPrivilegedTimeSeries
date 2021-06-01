
# Long-term Prediction with Privileged Time Series in Linear Dynamical Systems



This repository is the official implementation of [Long-term Prediction with Privileged Time Series in Linear Dynamical Systems](). 

### To-do list
- [ ] Create a requirements.txt file
- [x] Add a figure to describe the problem setup 
- [ ] Add executable experiment for synthetic
- [ ] Add executable experiment + data (link or dataset) for FC
- [ ] Add code for ADNI
- [ ] Add code for MM
- [ ] Add some main results in README.md

![](images/illustration-crop.png)

### Abstract
We study learning of predictive models of long-term outcomes that are given access to privileged information in the form of intermediate time series. These time series, available only in training samples, are observed chronologically between the baseline (time of prediction) and the outcome. We give an algorithm for this setting and prove that when the time series are drawn from a Gaussian-linear dynamical system, learning with privileged information is more efficient than learning without it. On synthetic data, we test the limits of our algorithm and theory,  both when our assumptions hold and when they are violated. On three diverse real-world datasets, we show that our approach is preferable to classical learning of predictive models in almost all settings, particularly when data is scarce.


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Evaluation

### Synthethic

To re-produce synthethic experiments, do the following:


>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

### Forecasting Air Quality


To re-produce the experiments on this real-world dataset, do the following:

The data is found in */data/fivecities/*, but can also be downloaded from [https://archive.ics.uci.edu/ml/datasets/PM2.5+Data+of+Five+Chinese+Cities](here).



### Modeling Progression of Chronic Disease

**Note**: For the Alzheimer’s and Multiple myeloma progression modeling tasks, the data is not publicly available, but the code which produced the results is still found in this repository. 

## Results

Our model achieves the following performance:

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Bibtex entry for citation

- [ ] To be filled out
