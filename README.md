
# Using Time-Series Privileged Information for Provably Efficient Learning of Prediction Models

### [Link to paper](https://arxiv.org/abs/2110.14993)

![](images/illustration-crop.png)

### Abstract
We study prediction of future outcomes with supervised models that use privileged information during learning. The privileged information comprises samples of time series observed between the baseline time of prediction and the future outcome; this information is only available at training time which differs from the traditional supervised learning. Our question is when using this privileged data leads to more sample-efficient learning of models that use only baseline data for predictions at test time.
We give an algorithm for this setting and prove that when the time series are drawn from a non-stationary Gaussian-linear dynamical system of fixed horizon, learning with privileged information is more efficient than learning without it.  On synthetic data, we test the limits of our algorithm and theory,  both when our assumptions hold and when they are violated. 
On three diverse real-world datasets, we show that our approach is generally preferable to classical learning, particularly when data is scarce. Finally, we relate our estimator to a distillation approach both theoretically and empirically.

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

