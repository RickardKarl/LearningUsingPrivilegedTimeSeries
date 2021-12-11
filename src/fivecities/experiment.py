"""
Experiment code used by fivecities.ipynb
"""

# Import packages and classes

import sys 
import os
sys.path.append('..')

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from itertools import product 

# Import packages
from src.model import baseline, lupts, distillation
from src.fivecities.reader import FiveCities
from src.plotutils import set_mpl_default_settings, method_color, method_marker, score_label

# Model
model_dict_default = {
    'Distill-Seq': distillation.DistillationModel(distill_seq=True),
    'Distill-Concat': distillation.DistillationModel(),
    'Baseline' : baseline.Baseline(),
    'LuPTS' : lupts.LUPTS()
#                 'Stat-LuPTS': lupts.StatLUPTS()
                }

# Experiment parameters
# default values
default_values = {
    'n_list': list(range(100,825,50)),
    'score' : r2_score,
    'iterations' : 75
}

# Model hyperparameters 
hp_args = {
    'Distill-Seq': {'lambda': [0.25, 0.5, 0.75], 'epochs': [200], 'lr': [1e0]}, 
    'Distill-Concat': {'lambda': [0.25, 0.5, 0.75], 'epochs': [200], 'lr': [1e0]}
}

# Initilize variables and experiment parameters
set_mpl_default_settings()

# save figures
save = True
save_folder = '../results/fivecities/'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

data_path = '../data/fivecities'


def run_experiment(city : str, sequence_length : int, timestep_list : list, n_list = default_values['n_list'], \
                   model_dict=model_dict_default, fc_args = {}):
    
    """
    city : select among ['beijing', 'shanghai', 'shenyang', 'chengdu','guangzhou']
    sequence_length : Outcome is sequence_length + 1 hours into the future
    timestep_list : Will decide which time steps to include as privileged information by picking indices according to range(0,sequence_length, timestep) for all timestep in timestep_list 
    n_list : what sample sizes to vary over
    fc_args : additional args to FiveCities class
    """

    fc_args = {**{'sequence_length': sequence_length, 'city_list' : [city]},  **fc_args}
    fc = FiveCities(data_path, args=fc_args)


    output_dict = {'city': city,
                    'score': default_values['score'],
                    'sequence_length' : sequence_length,
                    'timestep' : {},
                    'model_list' : list(model_dict.keys())}

    
    for timestep in timestep_list:
        
        saved_samples = []
        res_dict = {}
        for model in model_dict:
            res_dict[model] = []
        
        for n in n_list:
            
            # Skip if we want to sample more samples than there is available
            if n>fc.get_max_train_samples(city):
                continue
            
            saved_samples.append(n)
            tmp_res_dict = {}
            for model in model_dict:
                tmp_res_dict[model] = []


            for _ in range(default_values['iterations']):

                X_train, X_test, y_train, y_test  = fc.sample(city, sample_size=n, timestep=timestep)

                X_train, X_test = FiveCities.scaler(X_train, X_test, fc.continuous_var_index)
                X_train, X_test = FiveCities.mean_imputation(X_train, X_test)
                
                for model in model_dict:
                    # do a hp search over specified parameters (only supported for distillation models, e.g. Distillation)
                    if model in hp_args: 
                        # split up training set into train and validate
                        tr_idxs, val_idxs = train_test_split(np.arange(X_train.shape[0]),test_size=0.2,random_state=42)
                        X_hp_train = X_train[tr_idxs]; X_hp_valid = X_train[val_idxs]
                        y_hp_train = y_train[tr_idxs]; y_hp_valid = y_train[val_idxs]
                        
                        # do hp search and then refit with best combo 
                        param_names = hp_args[model].keys()
                        param_lists = [hp_args[model][k] for k in param_names]
                        
                        best_hp = (-1000, {})
                        for elem in product(*param_lists): 
                            if model == 'Distill-Seq':     
                                params = {k:elem[i] for i,k in enumerate(param_names)}
                                hp_model = distillation.DistillationModel(train_args=params, distill_seq=True)
                            elif model == 'Distill-Concat': 
                                params = {k:elem[i] for i,k in enumerate(param_names)}
                                hp_model = distillation.DistillationModel(train_args=params)
                            else: 
                                raise ValueError('No support for hp search over non distillation models.')
                            hp_model.fit(X_hp_train, y_hp_train)
                            new_r2 = default_values['score'](y_hp_valid, hp_model.predict(X_hp_valid))
                            if new_r2 > best_hp[0]: 
                                best_hp = (new_r2, params)
                        print(f'timestep: {timestep}, n: {n}, model: {model}, params: {best_hp}')
                        model_dict[model].set_train_args(best_hp[1])
                    # fit model
                    model_dict[model].fit(X_train, y_train)
                
                for model in model_dict:
                    tmp_res_dict[model].append(default_values['score'](y_test, model_dict[model].predict(X_test)))

            for model in model_dict:
                res_dict[model].append((np.mean(tmp_res_dict[model]), np.std(tmp_res_dict[model])))


        output_dict['timestep'][timestep] = { model : res_dict[model] for model in model_dict}

    output_dict['n_list'] = saved_samples
    return output_dict


"""
    Plotting code
"""

def plot_results_timehorizons(output_dict, include_only_model = None, include_only_timestep = [1], save=save, legend=True, xlim = None, ylim = None, title=''):
      
    n_list = output_dict['n_list']
    city = output_dict['city']
    score_function = output_dict['score']
    sequence_length = output_dict['sequence_length']
    
    for idx, timestep in enumerate(output_dict['timestep']):
        
        plt.figure(figsize=(7, 4.5), dpi=160)
        
        if include_only_model is None: model_list = output_dict['model_list']
        #if timestep not in include_only_timestep: continue

        for model in model_list:

            # If there is only one privileged time point, Stat-LuPTS is identical to LUPTS and do not need to be plotted
            if len(list(range(0, sequence_length, timestep))) == 2 and model == 'Stat-LuPTS':
                continue 

            # Read for a particular model
            mean = np.asarray([m[0] for m in output_dict['timestep'][timestep][model]])
            standard_dev = np.asarray([m[1] for m in output_dict['timestep'][timestep][model]])

            # Plot 
            plt.plot(n_list, mean, label=model, c=method_color(model), marker=method_marker(model), markevery=2)
            plt.fill_between(n_list, mean-standard_dev, mean+standard_dev,  color=method_color(model), alpha = 0.2)

        ## Customize plot
        # Change axis limits if desired
        if ylim: plt.ylim(ylim)
        if xlim: plt.xlim(xlim)

        plt.ylabel(score_label(score_function.__name__))
        plt.xlabel(f'Number of training samples')
        if legend:
            plt.legend(loc='lower right')
        plt.grid(True)
        plt.title(title)

        if save:
            path = os.path.join(save_folder, f'experiment_{city}_ts{timestep}_T{sequence_length}')
            plt.savefig(path+'.pdf', format='pdf', bbox_inches='tight')

def plot_results_PTS(output_dict, include_only_model = ['Baseline', 'LuPTS'], save=save, legend=True, xlim = None, ylim = None, title=''):
    
    """
    Make all privileged time amounts variants in the same figure
    """

    n_list = output_dict['n_list']
    city = output_dict['city']
    score_function = output_dict['score']
    sequence_length = output_dict['sequence_length']
    timestep_list = list(output_dict['timestep'].keys())
    
    plt.figure(figsize=(7, 4.5), dpi=160)

    plot_baseline_once = False

    for idx, timestep in enumerate(output_dict['timestep']):
        
        if include_only_model is None: 
            model_list = output_dict['model_list']
        else:
            model_list = include_only_model

        for model in model_list:

            # If there is only one privileged time point, Stat-LuPTS is identical to LUPTS and do not need to be plotted
            if len(list(range(0, sequence_length, timestep))) == 2 and model == 'Stat-LuPTS':
                continue 
            
            
            #Somehow just choose the best Distillation seq and concat respectively over timesteps
            #Basing this on the experiment with most available samples
            best_mean_score = -100000000
            time_to_keep = -1
            if model != 'Baseline' and model != 'LuPTS':
                for tstep in output_dict['timestep'].keys():
                    m = output_dict['timestep'][tstep][model][-1][0]
                    if m > best_mean_score:
                        best_mean_score = m
                        time_to_keep = tstep

            

            
            if (model != 'Baseline' and model != 'LuPTS') and (timestep != time_to_keep):
                continue
            


            # If we wish to skip a particular PTS value
            if timestep not in timestep_list:
                continue

   
            else:
                
                # Compute the number of privileged time points by skipping first index and then counting number of steps we perform
                if model == 'LuPTS':
                    
                    nbr_PTS = len(list(range(1, sequence_length, timestep))) 

                    # Hard-coded colors (only built for two different PTS, but easily fixed)
                    if timestep == 1:
                        color = 'orange'
                        label = model + f'_{nbr_PTS}PTS'
                        marker = 'p'
                    else:
                        color = 'black'
                        label = model + f'_{nbr_PTS}PTS'
                        marker = 'X'
                
                elif model == 'Distill-Concat':
                    nbr_PTS = len(list(range(1, sequence_length, timestep)))
                    color = 'green'
                    label = model + f'_{nbr_PTS}PTS'
                    marker = 'D'

                elif model == 'Distill-Seq':
                    nbr_PTS = len(list(range(1, sequence_length, timestep)))
                    color = 'purple'
                    label = model + f'_{nbr_PTS}PTS'
                    marker = '^'

                # We should only plot baseline once (we are looping over many experiments)
                elif plot_baseline_once:
                    continue
                else:
                    color = method_color(model)
                    label = model
                    marker = method_marker(model)
                    plot_baseline_once=True
                    
                # Get results
                mean = np.asarray([m[0] for m in output_dict['timestep'][timestep][model]])
                standard_dev = np.asarray([m[1] for m in output_dict['timestep'][timestep][model]])

                # Plot
                
                plt.plot(n_list, mean, label=label, c=color, marker=marker, markevery=2)
                plt.fill_between(n_list, mean-standard_dev, mean+standard_dev, color=color, alpha = 0.2)


        # Change axis limits if desired
        if ylim: plt.ylim(ylim)
        if xlim: plt.xlim(xlim)

        plt.ylabel(score_label(score_function.__name__))
        plt.xlabel(f'Number of training samples')
        if legend:
            plt.legend(loc='lower right')
        plt.grid(True)
        plt.title(title)
    
    if save:
        path = os.path.join(save_folder, f'experiment_{city}_varyPTS_T{sequence_length}')
        plt.savefig(path+'.pdf', format='pdf', bbox_inches='tight')