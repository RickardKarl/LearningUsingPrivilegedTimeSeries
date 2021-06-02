"""
Experiment code used by synthetic.ipynb
"""
import sys 
import os
sys.path.append('..')

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score, mean_squared_error

from src.model import baseline, lupts
from src.synthethic.generate import create_default_system
from src.plotutils import set_mpl_default_settings, method_color, method_marker, score_label


# Experiment parameters

# default values
default_values = {
    'sequence_length': 10,
    'num_samples' : 1000,
    'dim' : 25,
    'seed' : 42,
    'iterations' : 100,
    'noise_var' : 1,
    'input_var': 5,
    'spec_radius' : 1.5
}

# Model
model_dict = {'Baseline' : baseline.Baseline(),
                'LuPTS' : lupts.LUPTS(),
                'Stat-LuPTS': lupts.StatLUPTS()}

# Initilize variables and experiment parameters
set_mpl_default_settings()

# save figures
save = True
save_folder = '../results/synthetic'
if not os.path.exists(save_folder) and save:
    os.mkdir(save_folder)

# Relative MSE
relMSE = lambda real_coef, estimated_coef : np.linalg.norm(real_coef - estimated_coef) / np.linalg.norm(estimated_coef)


# Training code
def train_eval(system, model, numSamples = default_values['num_samples'], seqLength = default_values['sequence_length'], dim = default_values['dim']):

    args = {'system': system,
            'model': model,
            'numSamples': numSamples,
            'seqLength': seqLength,
            'scoring' : r2_score,
            'testFraction': 0.2,
            'seed': default_values['seed']
            }

    #Draw system sample and train/test split.
    X_train, y_train = args['system'].sample(args['numSamples'], args['seqLength'])
    X_test, y_test = args['system'].sample(args['numSamples'], args['seqLength'])
    
    #Fit model
    args['model'].fit(X_train, y_train) 
    test_error = args['scoring'](y_test, args['model'].predict(X_test))
    output_dict = {
        'model'  : args['model'],
        'test_error' : test_error
        }
    return output_dict

# Train/eval loop over a particular variable which is varied
def run_experiment(vary_list, label, system_method,
                        dim = default_values['dim'], 
                        sequence_length = default_values['sequence_length'],
                        num_samples = default_values['num_samples'],
                        noise_var = default_values['noise_var'],
                        input_var = default_values['input_var'],
                        spec_radius=default_values['spec_radius'],
                        beta2 = None,
                        stationary = False):


    param_recovery_dict = {} # parameter recovery
    test_error_dict = {} # test error
    for key in model_dict:
            param_recovery_dict[key] = []
            test_error_dict[key] = []

    # Decides which variable to vary over
    for variable in tqdm(vary_list):    
            
            if label == 'dim':
                    dim = variable 
            elif label == 'sequence_length':
                    sequence_length = variable
            elif label == 'num_samples':
                    num_samples = variable
            elif label == 'noise_var':
                    noise_var = variable
            elif label == 'beta2':
                    beta2 = variable
            elif label == 'spec_radius':
                    spec_radius = variable
            else:
                    raise ValueError('Invalid label argument')


            tmp_paramrec_dict = {} # parameter recovery
            tmp_test_error_dict = {} # test error
            for key in model_dict:
                    tmp_paramrec_dict[key] = []
                    tmp_test_error_dict[key] = []
            

            # Experiment loop per variable value
            for _ in range(default_values['iterations']):

                    # Generate system
                    system = system_method(dim, noise_variance=noise_var, beta2_scale=beta2, stationary=stationary, seq_length=sequence_length, input_variance=input_var, spec_radius=spec_radius)
                    # Compute true linear parameter
                    theta = system.real_linear_coefficient(sequence_length) 

                    # Train/eval
                    for key in model_dict:
                            # run
                            results_dict = train_eval(system, model_dict[key],num_samples, sequence_length)
                            tmp_score = relMSE(theta, results_dict['model'].get_estimator(sequence_length))
                            # Save for iteration
                            tmp_paramrec_dict[key].append(tmp_score)
                            tmp_test_error_dict[key].append(results_dict['test_error'])
            
            # Save results for variable value
            for key in model_dict:
                    param_recovery_dict[key].append((np.mean(tmp_paramrec_dict[key]), np.std(tmp_paramrec_dict[key])))
                    test_error_dict[key].append((np.mean(tmp_test_error_dict[key]), np.std(tmp_test_error_dict[key])))      

    # Save all results
    args = {}
    args['vary_list'] = vary_list
    args['label'] = label
    args['test_error'] = test_error_dict
    args['stationary'] = stationary

    param_recovery_dict['tag'] = 'parameter_recovery'
    test_error_dict['tag'] = 'test_error'

    return param_recovery_dict, test_error_dict, args


"""
Plotting code for results
"""

def plot_results(res_dict, args,
             test_error=False,
             xlabel=None,
             legend=True, 
             log=False,
             save=save, 
             save_label='', 
             exclude_stat = True):
    
    plt.figure(figsize=(4, 4), dpi=160)

    for key in model_dict:
        if key not in res_dict: continue
        if exclude_stat and key == 'Stat-LuPTS': continue

        # retrieve data and plot for a specific method
        mean = np.asarray([m[0] for m in res_dict[key]])
        standard_dev = np.asarray([m[1] for m in res_dict[key]])
        plt.plot(args['vary_list'], mean, label=key,  c=method_color(key), marker=method_marker(key), markevery = 2)
        plt.fill_between(args['vary_list'], mean-standard_dev, mean+standard_dev,  color=method_color(key), alpha = 0.2)
    

    # Select xlabel based on what we varied over
    if xlabel is None: 
        if args['label'] == 'beta2':
                xlabel = 'Ratio between $||\\delta||_2$ and $||\\beta||_2$'
        elif args['label'] == 'num_samples':
                xlabel = 'Number of training samples'
        elif args['label'] == 'dim':
                xlabel = 'Dimension'
        elif args['label'] == 'noise_var':
                xlabel = 'System noise variance'
        elif args['label'] == 'sequence_length':
                xlabel = 'Sequence length T' 
        elif args['label'] == 'spec_radius':
                xlabel = 'Spectral radius'
        else:
                xlabel = args['label']

    
    plt.xlabel(xlabel)
    plt.grid()

    # Customize plot
    if res_dict['tag'] == 'test_error':
        plt.ylabel(score_label('r2_score'))
    elif res_dict['tag'] == 'parameter_recovery':
        plt.ylabel(f'Relative MSE')
    else:
        raise ValueError('Unknown tag key in res_dict')
    
    if legend:
        plt.legend()
    if log:
        plt.yscale('log')
        plt.xscale('log')
    if save:
        save_label = args['label']
        tag = res_dict['tag']
        path = os.path.join(save_folder, f'experiment_{save_label}_{tag}')
        if log: path += '_log'
        if args['stationary']: path += '_stat'
        path += f'_{save_label}'
        plt.savefig(path+'.pdf', format='pdf',bbox_inches = "tight")