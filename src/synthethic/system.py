

import numpy as np
import functools
from .. import base


# A wrapper class for a numpy implemented distribution
class Distribution(base.Data):
    
    # distribution = {'dist' : np_function_handle,
    #         'dist_params' : [params1, params2, ...]
    #        }
    def __init__(self, distribution, sample_dim):
        self.distribution = distribution
        self.sample_dim = sample_dim
    
    def sample(self, shape):
        return self.distribution['dist'](*self.distribution['dist_params'], shape)
    
    #Does the distribution return scalar or vector.
    def dim(self):
        return self.sample_dim


class EasyA(base.ParamGen):

    def __init__(self, size, scale=1.0, seed = None, args={}):
        self.size = size
        self.scale = scale
        self.seed = seed
        self.single = False

    def gen(self):
        if not self.single:
            np.random.seed(self.seed)
            self.single=True
 
        matrix = np.identity(self.size) # ones along diag
        random_elements = np.random.normal(0, self.scale, (self.size,self.size)) # samples
        np.fill_diagonal(random_elements, 0) # set diag to zero
        return  matrix + random_elements # add

    def get_size(self):
        return self.size

class EasyBeta(base.ParamGen):

    #mask: Number of elements to set to zero.
    def __init__(self, size, scale = 1.0, mask=None, seed = None):
        self.size = size
        self.mask = mask
        self.seed = seed
        self.scale = scale
        self.single = False
    
    def gen(self):
        if not self.single:
            np.random.seed(self.seed)
            self.single=True

        beta = np.random.normal(0, self.scale, size=self.size)
        if self.mask is None:
            return beta
        mask_idx = np.random.choice(range(self.size), self.mask, replace=False)
        beta[mask_idx] = 0
        return beta

    def get_size(self):
        return self.size

class ProtSystem(base.Data):
    

    """
    parameters = {'A' : np.array / [np.array]) list of np.arrays A if non-stationary
                  'beta' : np.array,
                  'beta_2 : np.array} if break_markov = True
    """
    def __init__(self, data : Distribution,
                        system_noise : Distribution,
                        reg_noise : Distribution,
                        parameters : dict,
                        func = None, 
                        break_markov = False):
        
        if not isinstance(data, base.Data):
            raise Exception('data is not a member of Data')
        if not isinstance(system_noise, base.Data):
            raise Exception('system noise is not a member of Data')
        if not isinstance(system_noise, base.Data):
            raise Exception('regression noise is not a member of Data')
            
        self.data = data
        self.system_noise = system_noise # Might be worth to have two noise distributions
        self.reg_noise = reg_noise # Assumed to be dim 1
        self.parameters = parameters
        self.func = func
        self.break_markov = break_markov


    def real_linear_coefficient(self, seq_length : int):
        """
            Given a seq_length (int) it will return the true linear coefficient
        """
        if type(self.parameters['A']) == list:
            if len(self.parameters['A']) != seq_length-1:
                raise ValueError('Oppsie!')
            coef = np.matmul(functools.reduce(lambda A, B: np.matmul(A, B), self.parameters['A']), self.parameters['beta'].reshape(-1,1))
        else:    
            A = self.parameters['A']
            beta = self.parameters['beta'].reshape(-1,1)
            coef = np.matmul(np.linalg.matrix_power(A, seq_length-1), beta)

        if self.break_markov:
            if 'beta_2' in self.parameters.keys():
                # Add the direct link between X1 and Y
                coef = coef + self.parameters['beta_2'].reshape(-1,1)

        return coef
    
    def sample(self, samples : int, seq_length : int):
        
        if type(self.parameters['A']) == list:
            if len(self.parameters['A']) != seq_length-1:
                raise ValueError('Oppsie!')
            else:
                dim = len(self.parameters['A'][0])
        else:
            dim = len(self.parameters['A'])

        X = np.zeros((samples, seq_length, dim))
        #Check if input distribution generate vectors or scalars
        if self.data.dim() == 1:
            X[:,0,:] = self.data.sample((samples, dim))
        else:
            X[:,0,:] = self.data.sample(samples)

        for seq in range(1, seq_length):
            for sample in range(0, samples):

                # Apply transition
                if self.func is None:
                    if type(self.parameters['A']) != list:
                        X[sample,seq,:] = np.matmul(X[sample,seq-1,:], self.parameters['A'])
                    else:
                        X[sample,seq,:] = np.matmul(X[sample,seq-1,:], self.parameters['A'][seq-1]) 
                else:
                    if type(self.parameters['A']) != list:
                        X[sample,seq,:] = np.array(list(map(self.func, np.matmul(X[sample,seq-1,:], self.parameters['A']))))
                    else:
                        X[sample,seq,:] = np.array(list(map(self.func, np.matmul(X[sample,seq-1,:], self.parameters['A'][seq-1]))))

                # Add noise
                if self.data.dim() == 1:
                    epsilon = self.system_noise.sample(dim)
                else:
                    epsilon = self.system_noise.sample(1)
                    epsilon = np.reshape(epsilon, (-1,))
                X[sample,seq] += epsilon
        if self.break_markov:
            if 'beta_2' in self.parameters.keys():
                y = np.matmul(X[:,-1,:], self.parameters['beta']) + \
                    np.matmul(X[:,0,:], self.parameters['beta_2'])
            else:
                raise ValueError('Provide beta_2 in parameters')
        else:
            y = np.matmul(X[:,-1,:], self.parameters['beta'])

        y += self.reg_noise.sample(X.shape[0])

        return X, y
    
    def seed(self):
        pass
    
    def describe(self):
        pass


