from .. import base
from . import modelutils

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV

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

    def __init__(self, cv_search = False, folds = 5, random_state = None, logistic_args={}):

        if cv_search:
            self.regression_model = LogisticRegressionCV(solver='liblinear', n_jobs=-1, max_iter=1000, scoring='roc_auc', cv=folds, random_state=random_state)
        else:
            self.regression_model = LogisticRegression(**logistic_args)
    
    def fit(self, X : np.array, y : np.array):
        self.regression_model.fit(X[:,0,:], y)

    def predict(self, X : np.array) -> np.array:
        return self.regression_model.predict(X[:,0,:])

    def predict_proba(self, X : np.array) -> np.array:

        return self.regression_model.predict_proba(X[:,0,:])


class RFRegressorLuPTS():

    """
    Adaption of RandomForestRegressor from sklearn with built-in hyperparameter opt.
    """
    def __init__(self):

        self.model = RandomForestRegressor()
        self.rfCV = None

        # Hyperparameters for regressor

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        self.random_grid = {'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

    def fit(self, X : np.array, y : np.array):

        self.rfCV = RandomizedSearchCV(estimator=self.model,
                                        param_distributions=self.random_grid,
                                        n_jobs=-1,
                                        n_iter=50,
                                        cv=2)
        self.rfCV.fit(X[:,0,:], y)
    
    def predict(self, X : np.array) -> np.array:
        return self.rfCV.predict(X[:,0,:])


class KNRegressorLuPTS():

    """
    Adaption of KNeighborsRegressor from sklearn with built-in hyperparameter opt.
    """
    def __init__(self):

        self.model = KNeighborsRegressor()
        self.knCV = None

        # Hyperparameters for regressor

        # Number of neighbors
        n_neighbors = [int(x) for x in np.linspace(start = 1, stop = 25, num = 5)]
        # Weight function
        weights = ['uniform', 'distance']
        # Leaf_size
        leaf_size = [5, 10, 30]
        # Power parameter
        p = [1,2]
        # Create the random grid
        self.random_grid = {'n_neighbors': n_neighbors,
                    'weights': weights,
                    'leaf_size': leaf_size,
                    'p': p}

    def fit(self, X : np.array, y : np.array):

        self.knCV = RandomizedSearchCV(estimator=self.model,
                                        param_distributions=self.random_grid,
                                        n_jobs=-1,
                                        n_iter=50,
                                        cv=2)
        self.knCV.fit(X[:,0,:], y)
    
    def predict(self, X : np.array) -> np.array:
        return self.knCV.predict(X[:,0,:])


class MLPRegressorLuPTS():

    """
    Adaption of MLPRegressor from sklearn with built-in hyperparameter opt.
    """
    def __init__(self):

        self.model = MLPRegressor(activation='tanh', solver ='lbfgs')
        self.mlpCV = None

        # Hyperparameters for regressor

        # Number of neurons
        hidden_layer_sizes = [(i,) for i in range(10,100,10)]
        # L2 reg.
        alpha = [0.0001, 0.001, 0.01]
        # Create the random grid
        self.random_grid = {'hidden_layer_sizes': hidden_layer_sizes,
                    'alpha': alpha
                    }

    def fit(self, X : np.array, y : np.array):

        # Mute warnings from sklearn due to MLPRegressor not converging
        def warn(*args, **kwargs):
            pass
        import warnings
        warnings.warn = warn

        self.mlpCV = RandomizedSearchCV(estimator=self.model,
                                        param_distributions=self.random_grid,
                                        n_jobs=-1,
                                        n_iter=50,
                                        cv=2)
        self.mlpCV.fit(X[:,0,:], y)
    
    def predict(self, X : np.array) -> np.array:
        return self.mlpCV.predict(X[:,0,:])


class ModelWrapper():

    '''
    Wrapper for any model
    '''

    def __init__(self, model):
        self.model = model 

    def fit(self, X, y):
        self.model.fit(X[:,0,:], y)
    
    def predict(self, X):
        return self.model.predict(X[:,0,:])

    def predict_proba(self, X):
        return self.model.predict_proba(X[:,0,:])
