
from .. import base
from . import modelutils

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score


# Shape of X: [sample, timesteps, dimension]

class StatLUPTS(base.Model):
    """
        Linear regression model using privileged information from A to learn predictive model of outcome Y
        Stationary variant
    """
    def __init__(self, args={}, model_args={}):
        """
        input args 
            'reg' : (str) Type of regularisation. Default is None
            'intercept' : (bool) fit intercept for dynamical system X_1 -> X_2 -> ... -> X_T. Default is False
            'intercept_last_step' : (bool) fit intercept for last step regression X_T->Y. Default is True

        input model_args 
            see possible arguments for LinearRegression(), Lasso() and similar method in sklearn
        """
        self.args = args           # for general args
        self.model_args=model_args # for model args

        self.system_model = None

        intercept = args.get('intercept_last_step') if 'intercept_last_step' in args else True
        self.regression_model = modelutils.linear_model(self.args, model_args=self.model_args, intercept=intercept)


    def fit(self, X : np.array, y : np.array):
        
        self.system_model = modelutils.approxA(X, self.args, self.model_args)
        self.regression_model.fit(X[:,-1,:], y)

    def get_predictor(self, X : np.array) -> np.array:

        _, nbr_timesteps, dim = X.shape
        
        if dim > 1:
            X0 = X[:,0,:].squeeze()
        else:
            X0 = X[:,0,:].reshape((-1,1))

        transition = np.linalg.matrix_power(self.system_model, nbr_timesteps-1)
        
        predictor = np.matmul(X0, transition)
        return predictor

    def predict(self, X : np.array) -> np.array:

        return self.regression_model.predict(self.get_predictor(X))

    def plot_rollout(self, inputX : np.array, seqLength = None, sample_idx = 0):

        rollout = self.predict_rollout(inputX, seqLength)

        prediction = self.regression_model.predict(rollout[:,-1,:])

        plot_trajectory_from_data(rollout, prediction, sample_idx)

    def predict_rollout(self, inputX : np.array, seqLength = None):
        
        n_sample, _ , dim = inputX.shape
        seqLength = inputX.shape[1] if seqLength is None else seqLength

        rollout = np.zeros((n_sample, seqLength, dim)) 

        rollout[:,0,:] = inputX[:, 0, :]
        
        for i in range(1, seqLength):

            rollout[:,i,:] = np.matmul(rollout[:,i-1,:], self.system_model)
        
        return rollout

    def get_estimator(self, nbr_timesteps : int) -> np.array:
        """ 
        Returns the estimation of A^{T-1}*beta
        """
        
        coef = self.regression_model.coef_.reshape(-1, 1)
        return np.matmul(np.linalg.matrix_power(self.system_model, nbr_timesteps-1), coef)



class LogisticLStatLUPTS(StatLUPTS):
    """
        StatLUPTS for classification.
        Logistic regressiona at the step X_T -> Y
    """
        
    def __init__(self, cv_search = False, folds = 5, args={}, model_args={}, logistic_args={}):

        super().__init__(args=args, model_args=model_args)
        if cv_search:
            self.regression_model = LogisticRegressionCV(solver='liblinear', n_jobs=-1, max_iter=1000, scoring='roc_auc', cv=folds)
        else:    
            self.regression_model = LogisticRegression(**logistic_args)


    def fit(self, X : np.array, y : np.array):
        
        self.system_model = modelutils.approxA(X, self.args, self.model_args)
        self.regression_model.fit(X[:,-1,:], y)

    def predict_proba(self, X : np.array) -> np.array:

        return self.regression_model.predict_proba(self.get_predictor(X))
    
    def get_estimator(self, nbr_timesteps : int ) -> np.array:
        pass 



class LUPTS(StatLUPTS):

    def __init__(self, args={}, model_args={}):

        """
        Non-stationary LUPTS for regression.
        Learns different functions for the transition between any pairs of X_{t-1} -> X_{t} for t=2,...,T
        """
        
        super().__init__(args=args, model_args=model_args)
        self.system_model = []

    def fit(self, X : np.array, y : np.array):

        self.system_model = []
        intercept = self.args.get('intercept') if 'intercept' in self.args else False
        nbr_transitions= X.shape[1]

        for timestep in range(nbr_transitions-1):
            model = modelutils.linear_model(self.args, model_args=self.model_args, intercept=intercept)
            model.fit(X[:,timestep,:], X[:,timestep+1,:])
            self.system_model.append(model)

        self.regression_model.fit(X[:,-1,:], y)

    def get_predictor(self, X : np.array):

        _, nbr_timesteps, dim = X.shape
        
        if dim > 1:
            X0 = X[:,0,:].squeeze()
        else:
            X0 = X[:,0,:].reshape((-1,1))

        # Applies each linear model per time step in order
        predictor = X0
        for model in self.system_model:
            predictor = model.predict(predictor)

        return predictor

    def predict(self, X : np.array) -> np.array:
        return self.regression_model.predict(self.get_predictor(X))

    def plot_rollout(self, inputX : np.array, seqLength = None, sample_idx = 0):

        rollout = self.predict_rollout(inputX, seqLength)

        prediction = self.regression_model.predict(rollout[:,-1,:])

        plot_trajectory_from_data(rollout, prediction, sample_idx)

    def predict_rollout(self, inputX : np.array, seqLength = None):
        
        n_sample, _ , dim = inputX.shape
        seqLength = inputX.shape[1] if seqLength is None else seqLength

        rollout = np.zeros((n_sample, seqLength, dim)) 

        rollout[:,0,:] = inputX[:, 0, :]
        
        for i, model in enumerate(self.system_model):

            rollout[:,i+1,:] = model.predict(rollout[:,i,:])
        
        return rollout

    def get_estimator(self, nbr_timesteps : int) -> np.array:
        
        estimator = self.system_model[0].coef_.T

        for model in self.system_model[1:]:
            estimator = np.matmul(estimator, model.coef_.T)

        coef = self.regression_model.coef_.reshape(-1, 1)
        estimator = np.matmul(estimator, coef)
        return estimator

class LogisticLUPTS(LUPTS):

    def __init__(self, cv_search=False, folds = 5, args={}, model_args={}, logistic_args={}):
                
        """
        Non-stationary LUPTS for classification.
        Learns different functions for the transition between any pairs of X_{t-1} -> X_{t} for t=2,...,T
        """
        
        super().__init__(args=args, model_args=model_args)
        if cv_search:
            self.regression_model = LogisticRegressionCV(solver='liblinear', n_jobs=-1, max_iter=1000, scoring='roc_auc', cv = folds)
        else:    
            self.regression_model = LogisticRegression(**logistic_args)

    def predict_proba(self, X : np.array) -> np.array:

        predictor = self.get_predictor(X)
        return self.regression_model.predict_proba(predictor)

    def get_estimator(self, nbr_timesteps : int):
        pass

    
