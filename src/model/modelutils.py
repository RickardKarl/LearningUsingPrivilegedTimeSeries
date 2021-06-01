
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso

def linear_model(args, model_args = {}, intercept = False):
    """
    Returns linear model depending on args
    """

    reg_method = args.get('reg') if ('reg' in args) else None 

    if reg_method == 'lasso':
        return Lasso(fit_intercept=intercept, **model_args)
    elif reg_method == 'ridge':
        return Ridge(fit_intercept=intercept, **model_args)
    else:
        return LinearRegression(fit_intercept=intercept)

def approxA(X, args : dict, model_args : dict) -> np.array:

    """
    Used by stationary LuPTS
    """

    dim = X.shape[2]

    system_transition = np.zeros((dim,dim)) # Create matrix for coefficients
    
    predictor = X[:,:-1,:].reshape((-1,1,dim))
    response = X[:,1:,:].reshape((-1,1,dim))
        
    predictor = predictor.squeeze()
    response = response.squeeze()

    # Fit model
    intercept = args.get('intercept') if 'intercept' in args else False
    model = linear_model(args, model_args=model_args, intercept=intercept).fit(predictor, response)
    system_transition = model.coef_.T # Save coefficients of linear model into matrix

    return system_transition