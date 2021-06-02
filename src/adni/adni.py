import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.impute import SimpleImputer
import functools
import pandas as pd
import math

from ..model.lupts import LUPTS, StatLUPTS, LogisticLUPTS, LogisticStatLUPTS
from ..model.baseline import Baseline, LogisticBaseline
from  ..plotutils import method_color, method_marker, set_mpl_default_settings




cols_sel = ['MMSE', 'PTGENDER', 'APOE4', 'AGE', 'PTEDUCAT', 'FDG',
                'ABETA', 'TAU', 'PTAU', 'CDRSB', 'ADAS11', 'ADAS13', 'ADASQ4', 'RAVLT_immediate', 
                'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting', 'LDELTOTAL',  
                'TRABSCOR', 'FAQ', 'MOCA', 'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan', 
                'EcogPtOrgan', 'EcogPtDivatt', 'EcogPtTotal', 'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat', 
                'EcogSPPlan', 'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal', 
                'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV']

cols_categorical = ['PTGENDER', 'APOE4']



def quickADNI(set_path, task, priv_points, nan_threshold = 0.7, seed = 42):

   
    if task == 'MCIAD' or task == 'AD':
        target = 'AD'
    elif task == 'CNMCI':
        target = 'MCI'
    elif task == 'MMSE':
        target = 'MMSE'

    data_viscodes = ['bl', 'm12', 'm24', 'm36', 'm48']
    if priv_points == 1:
        selection_viscodes = ['bl', 'm24', 'm48']
    elif priv_points == 3:
        selection_viscodes = data_viscodes
    else:
        raise ValueError('priv_points invalid value: ' + str(priv_points))

    D = pd.read_csv(set_path)

    D['AD'] = D['DX']=='Dementia'
    D['MCI'] = D['DX']=='MCI'
    D.loc[D['DX'].isna(), ['AD', 'MCI']] = np.nan

    D.loc[:,'ABETA'] = D.loc[:,'ABETA'].replace('>1700', 1700, regex=True) \
                            .replace('<900', 900, regex=True) \
                            .replace('<200', 200, regex=True).astype(np.float32)

    D.loc[:,'TAU'] = D.loc[:,'TAU'].replace('>1300', 1300, regex=True) \
                        .replace('<80', 80, regex=True).astype(np.float32)

    D.loc[:,'PTAU'] = D.loc[:,'PTAU'].replace('>120', 120, regex=True) \
                        .replace('<8', 8, regex=True).astype(np.float32)

    D = D.loc[:,['VISCODE', 'RID', 'MCI', 'AD'] + cols_sel]
    D = pd.get_dummies(D, columns=cols_categorical)

    to_be_removed = []
    for code in data_viscodes:
        count = len(D[D['VISCODE'] == code])
        l = D[D['VISCODE'] == code].isna().sum()
        for i, e in enumerate(l):
            if nan_threshold < e/count:
                if D.columns[i] not in to_be_removed:
                    to_be_removed += [D.columns[i]]
    D = D.drop(to_be_removed, axis=1)

    frames = {}
    for code in data_viscodes:
        if code == data_viscodes[-1]:
            frames[code] = D[D['VISCODE'] == code].dropna(subset=[target])
        else:
            frames[code] = D[D['VISCODE'] == code]

    I = get_rids(frames, task, data_viscodes)

    data = {}
    for code in selection_viscodes:
        data[code] = frames[code][frames[code]['RID'].isin(I)]

    print(task)
    if task != 'MMSE':
        print('Number of subjects: '+str(len(I)))
        print('Number of positives at last time step: '+str(len(data[selection_viscodes[-1]][data[selection_viscodes[-1]][target] == 1].index)))
        print('Number of negatives at last time step: '+str(len(data[selection_viscodes[-1]][data[selection_viscodes[-1]][target] == 0].index)))
    else:
        print('Number of subjects: '+str(len(I)))

    features = [e for e in D.columns if e not in ['RID', 'VISCODE', 'MCI', 'AD']]

    X = np.zeros((len(I), len(selection_viscodes)-1, len(features)))
    data[selection_viscodes[-1]] = data[selection_viscodes[-1]].sort_values(by=['RID'])
    Y = data[selection_viscodes[-1]][target].values

    feature_index = {}

    for j, code in enumerate(selection_viscodes[0:len(selection_viscodes)-1]):
        data[code] = data[code].sort_values(by=['RID'])
        data[code] = data[code].loc[:,features]
        for feature in features:
            feature_index[feature] = data[code].columns.get_loc(feature)
        X[:,j,:] = data[code].values

    data_size = len(X)

    models = {}

    if task != 'MMSE':
        models['Baseline'] = LogisticBaseline(cv_search=True, folds=5, random_state = seed)
        models['LuPTS'] = LogisticLUPTS(cv_search=True, folds=5, random_state = seed)
        if priv_points == 3:
            models['Stat-LuPTS'] = LogisticStatLUPTS(cv_search=True, folds=5, random_state = seed)
    else:
        models['Baseline'] = Baseline()
        models['LuPTS'] = LUPTS()
        if priv_points == 3:
            models['Stat-LuPTS'] = StatLUPTS()

    step = 20
    bottom = 80
    top = math.floor(data_size*0.5)
    top = top - (top % step)

    tr_sample_sizes = range(bottom, top, step)

    results = {}

    np.random.seed(seed)
    rkf = RepeatedKFold(n_splits=2, n_repeats=50, random_state=seed)

    #Main loop
    for sample_size in tr_sample_sizes:
        
        results[sample_size] = {}
        tmp_results = {}
        for model_key in models.keys():
            tmp_results[model_key] = []

        #Splits, 2x50
        for i, (I_tr, I_ts) in enumerate(rkf.split(X)):

            sampled_I_tr = np.random.choice(I_tr, sample_size, replace=False)
            training_data = X[sampled_I_tr,:,:].copy()
            test_data = X[I_ts,:,:].copy()

            for ixx, code in enumerate(selection_viscodes[0:len(selection_viscodes)-1]):
                for j in range(training_data.shape[2]):
                    if all(np.isnan(training_data[:,ixx,j])):
                        print(j)
                        training_data[:,ixx,j] = np.mean(training_data[:,ixx-1,j])
                imputer = SimpleImputer()
                training_data[:,ixx,:] = imputer.fit_transform(training_data[:,ixx,:])
                if ixx == 0:
                    test_data[:,ixx,:] = imputer.transform(test_data[:,ixx,:])

            l_training_data = training_data.copy()
            l_test_data = test_data.copy()

            scaler = RobustScaler()
            lupi_scaler = RobustScaler()

            #Scale data for baseline
            training_data[:,0,:] = scaler.fit_transform(training_data[:,0,:])
            test_data[:,0,:] = scaler.transform(test_data[:,0,:])

            #Scale data for LuPTS models, using observations over all time points per feature.
            l_training_data = lupi_scaler.fit_transform(l_training_data.\
                            reshape((-1,X.shape[2]))).reshape((len(l_training_data), X.shape[1], X.shape[2]))
            l_test_data= lupi_scaler.transform(l_test_data.reshape((-1,X.shape[2])))\
            .reshape((len(I_ts), X.shape[1], X.shape[2]))

            #Fit and evaluate models
            for model_key in models.keys():

                if (model_key == 'LuPTS') or (model_key == 'Stat-LuPTS'):
                    models[model_key].fit(l_training_data, Y[sampled_I_tr])
                else:
                    models[model_key].fit(training_data, Y[sampled_I_tr])

                if task != 'MMSE':
                    if (model_key == 'LuPTS') or (model_key == 'Stat-LuPTS'):
                        tmp_results[model_key] += [roc_auc_score(Y[I_ts], models[model_key].predict_proba(l_test_data)[:,1])]
                    else:
                        tmp_results[model_key] += [roc_auc_score(Y[I_ts], models[model_key].predict_proba(test_data)[:,1])]
                else:
                    if (model_key == 'LuPTS') or (model_key == 'Stat-LuPTS'):
                        tmp_results[model_key] += [r2_score(Y[I_ts], models[model_key].predict(l_test_data))]
                    else:
                        tmp_results[model_key] += [r2_score(Y[I_ts], models[model_key].predict(test_data))]

        #Record results over iterations
        for model_key in models.keys():
            results[sample_size][model_key] = [np.mean(tmp_results[model_key]), np.std(tmp_results[model_key])]

    return results


def get_rids(frames, task, codes):
    
    if task == 'AD' or task == 'MMSE':
        #Tidigare kod. Lös det med lib
        pass
    
    #Maybe just paste this in subject selection as an if clause for cn->mci etc...
    elif task == 'CNMCI':
        #Select patients with a negative AD diagnosis at last time step
        #Select patients with CN status at baseline.
        frames[codes[-1]] = frames[codes[-1]][(frames[codes[-1]]['AD'] == 0)]
        frames[codes[0]] = frames[codes[0]][((frames[codes[0]]['MCI'] == 0) & (frames[codes[0]]['AD'] == 0))]
    
    elif task == 'MCIAD':
        #Select patients who are NOT CN at last time step.
        #Select patients with MCI at baseline.
        frames[codes[-1]] = frames[codes[-1]][((frames[codes[-1]]['AD'] == 1) | (frames[codes[-1]]['MCI'] == 1))]
        frames[codes[0]] = frames[codes[0]][((frames[codes[0]]['MCI'] == 1) & (frames[codes[0]]['AD'] == 0))]
        
    patient_ID = {}
    for code in codes:
        patient_ID[code] = frames[code]['RID'].unique()
        
    I = functools.reduce(lambda a, b: np.intersect1d(a, b), [patient_ID[k] for k in patient_ID.keys()])
    
    return I

def plot_result_dict(results, ylabel, title):
    
    set_mpl_default_settings()

    fig = plt.figure(figsize=(6,6))

    outer_keys = list(results.keys())
    model_keys = list(results[outer_keys[0]].keys())
    
    for model in model_keys:
        mean = np.array([results[size][model][0] for size in outer_keys])
        std = np.array([results[size][model][1] for size in outer_keys])
        
        plt.plot(outer_keys, mean, color=method_color(model), marker=method_marker(model))
        plt.fill_between(outer_keys, mean-std, mean+std, color=method_color(model), alpha=0.2)
    
    plt.xlabel('Number of training samples')
    plt.ylabel(ylabel)
    plt.grid()
    plt.title(title)
    
    plt.legend(model_keys)
    
    return fig

