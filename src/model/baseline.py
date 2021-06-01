from .. import base
from . import modelutils

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score

# Shape of X: [sample, timesteps, dimension]

# These models are simply implementations of the sklearn standard methods, but they have been wrapped
# to suit for input with a different dimensionality so that the same variables can be
# given to both baseline and LUPTS 

class Baseline(base.Model):

    """
        OLS linear regression model using the first time step in X as predictor to estimate outcome Y
    """

    def __init__(self, args={}, model_args={}):
        """
        input args 
            'reg' : (str) Type of regularisation. Default is None
        """
        # Params
        # See linear_model(self) for how to change reg and reg_alpha
        self.args = args
        self.model_args = model_args

        # Create model
        self.regression_model = modelutils.linear_model(self.args, model_args=self.model_args, intercept=True)

    def fit(self, X : np.array, y : np.array):
        self.regression_model.fit(X[:,0,:], y)


    def predict(self, X : np.array) -> np.array:
        return self.regression_model.predict(X[:,0,:])


    def get_estimator(self, nbr_timesteps=None) -> np.array:
        """
        Returns the estimator
        nbr_timesteps is not used
        """
        return self.regression_model.coef_.reshape(-1,1)

    def get_params(self, deep=True):
        return self.regression_model.get_params(deep)


class LogisticBaseline(base.Model):


    """
        Logistic  regression model using the first time step in X as predictor to estimate outcome Y
    """

    def __init__(self, cv_search = False, folds = 5, logistic_args={}):

        if cv_search:
            self.regression_model = LogisticRegressionCV(solver='liblinear', n_jobs=-1, max_iter=1000, scoring='roc_auc', cv=folds)
        else:
            self.regression_model = LogisticRegression(**logistic_args)
    
    def fit(self, X : np.array, y : np.array):
        self.regression_model.fit(X[:,0,:], y)

    def predict(self, X : np.array) -> np.array:
        return self.regression_model.predict(X[:,0,:])

    def predict_proba(self, X : np.array) -> np.array:

        return self.regression_model.predict_proba(X[:,0,:])
