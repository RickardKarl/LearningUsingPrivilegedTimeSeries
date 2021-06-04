import os
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer



def epoch_timestamp(year, month=1, day=1,hour=1):
    return datetime(year=year, month=month, day=day, hour=hour).timestamp()
    
class FiveCities():
    
    """
    Class for pre-processing and sampling of the FiveCities dataset
    """

    # List of cities
    city_list = ['beijing', 'shanghai', 'shenyang', 'chengdu','guangzhou']
    
    # Lists with targets 
    PM_dict = {'beijing': ['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan', 'PM_US Post'],
               'chengdu': ['PM_Caotangsi', 'PM_Shahepu', 'PM_US Post'],
               'guangzhou': ['PM_City Station', 'PM_5th Middle School', 'PM_US Post'],
               'shanghai': ['PM_Jingan', 'PM_US Post', 'PM_Xuhui'],
               'shenyang': ['PM_Taiyuanjie', 'PM_US Post', 'PM_Xiaoheyan']
              }
    
    feature_list = ['season', 'DEWP', 'HUMI', 'PRES', 'TEMP', 'cbwd', 'Iws']
    # Not included due to many outliers but not providing much predictive information:  ['precipitation', 'Iprec']
    
    categorical_list = ['season', 'cbwd']
    
    default_args = {
        'sequence_length' : 12,
        'gap' : 6,
        'target' : 'PM_US Post',
        'city_list' : city_list,
        'test_portion' : 0.2,
        'scale' : False,
        'mean_imputation' : False
        }
        # mean_imputation and scale is done in the training loop instead of in this class

    def __init__(self, path : str, args = {}):
        
        # Get default arguments if necessary
        for key in FiveCities.default_args:
            if key not in args:
                args[key] = FiveCities.default_args[key]
        self.args = args
        
        # Read data
        self.df_dict = {}
        self.train_data = {}
        self.test_data = {}
        self.header = {}
        convert_to_timestamp = lambda row: epoch_timestamp(int(row['year']), month=int(row['month']), day=int(row['day']), hour=int(row['hour']))
        
        #print("Pre-processing city data")
        for c in self.args['city_list']:
            
            # Read data from file 
            self.df_dict[c] = pd.read_csv(os.path.join(path, f'{c}.csv'))
            
            # Create a epoch timestamp which summarize data from year, month, day and hour columns
            # Not necessary
            self.df_dict[c]['timestamp'] = self.df_dict[c].apply(convert_to_timestamp,axis=1)
            
            
            # Subset with relevant features
            self.df_dict[c] = self.df_dict[c][(FiveCities.feature_list + [self.args['target']]+['timestamp'])]  # timestamp will be removed later in filter_dd method
            
            # Dummy encoding of categorical variables
            self.df_dict[c] = pd.get_dummies(self.df_dict[c], columns=FiveCities.categorical_list)

            # Exception for Beijing since it is missing a value for the cwbd categorical feature
            if c == 'beijing':
                self.df_dict[c]['cbwd_SW'] = 0

        
            # Find sequences of desired length in data without missing values
            # Return training and test set
            X_train, X_test, y_train, y_test = self.filter_df(self.df_dict[c], c)
            
            self.continuous_var_index = [self.header[c].index(feature) for feature in (FiveCities.feature_list + [self.args['target']]) if feature not in FiveCities.categorical_list]
            
            # Scaling 
            if self.args['scale']:
                X_train, X_test = FiveCities.scaler(X_train, X_test, self.continuous_var_index)


            # Imputation
            if self.args['mean_imputation']:
                X_train, X_test = FiveCities.mean_imputation(X_train, X_test)
    
 

            # Save data in dicts
            self.train_data[c] = (X_train, y_train)
            self.test_data[c] = (X_test, y_test)

            
        #print("Done!")
        
    def mean_imputation(X_train : np.array, X_test : np.array) -> (np.array, np.array):

        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        for t in range(X_train.shape[1]):
            X_train[:,t,:] = imputer.fit_transform(X_train[:,t,:])
            X_test[:,t,:] = imputer.transform(X_test[:,t,:])

        return X_train, X_test

    def scaler(X_train : np.array, X_test : np.array, cont_var_index : list) -> (np.array, np.array):

        scaler = RobustScaler(unit_variance=1)
        scaler.fit(X_train[:,0,cont_var_index])
        for t in range(X_train.shape[1]):
            X_train[:,t,cont_var_index] = scaler.transform(X_train[:,t,cont_var_index])
            X_test[:,t,cont_var_index] = scaler.transform(X_test[:,t,cont_var_index])
        
        return X_train, X_test

    def get_max_train_samples(self, city : str) -> int:
        return self.train_data[city][0].shape[0]
    
    def get_max_test_samples(self, city : str) -> int:
        return self.test_data[city][0].shape[0]
    
    def sample(self, city : str, sample_size = None, timestep=None) -> (np.array, np.array, np.array, np.array):
        """
            Sample from a choosen city
        """
        
        if city not in self.args['city_list']:
            raise ValueError('City invalid')
            
        X_train, y_train = self.train_data[city] 
        X_test, y_test = self.test_data[city]
        
        # Select samples if not all
        if sample_size:
            if sample_size > len(y_train):
                raise ValueError(f'Invalid sample size {sample_size} larger than available training samples {len(y_train)}')

            random_mask = np.random.choice(len(y_train), sample_size, replace=False)
            X_train = X_train[random_mask,:,:]
            y_train = y_train[random_mask]
        
        # Splice data with different larger time steps time
        if timestep:
            
            _, seq_length, _ = X_train.shape
            ts = list(range(0, seq_length, timestep))
            assert len(ts) >= 2, f'Invalid timestep {timestep} too large with sequence length {seq_length}'
            
            ts = np.array(ts, dtype=np.int16)
            X_train = X_train[:,ts,:]
            X_test = X_test[:,ts,:]
            
        return X_train, X_test, y_train, y_test
            
    def filter_df(self, df : pd.DataFrame, city) -> (np.array, np.array):
        """
            Filter dataframe (df) and return a train/test set 
        """
        idx = 0
        idx_list = []
        tmp_idx_list = []
        prev_timestamp = df.iloc[0]['timestamp'] - 3600  # initalize


        # Finds sequences with the desired length where there are no missing values
        # Adds a small gap between all sequences to decrease interdependence between sequences
        while idx <  len(df):

            data = df.iloc[idx][self.args['target']]
            time_diff =  df.iloc[idx]['timestamp'] - prev_timestamp 
            prev_timestamp = df.iloc[idx]['timestamp']

            if np.abs(time_diff - 3600) < 1e-2:
                
                if np.isnan(data):
                    if len(tmp_idx_list) >= self.args['sequence_length'] + 1:  # plus one since the outcome should come afterwards
                        idx_list.append(tmp_idx_list)
                    tmp_idx_list = [] # reset
                    idx += self.args['gap'] - 1
                else:
                    if len(tmp_idx_list) >= self.args['sequence_length'] + 1: # plus one since the outcome should come afterwards
                        idx_list.append(tmp_idx_list)
                        tmp_idx_list = [] # reset
                        idx += self.args['gap'] - 1
                    else:
                        tmp_idx_list.append(idx)
            else:
                # happens if the time difference between previous and current datapoint is not an hour exactly
                tmp_idx_list = [] # reset

            idx += 1
            
        df = df.drop(labels=['timestamp'], axis=1) # do not include timestamp in final features
        self.header[city] = list(self.df_dict[city].columns)

        if len(tmp_idx_list) >= self.args['sequence_length'] + 1:
            idx_list.append(tmp_idx_list)
        
        
        # Split data into training and test set
        split_idx = int(len(idx_list)*(1-self.args['test_portion'])) 
        train_list = idx_list[:split_idx]
        test_list = idx_list[split_idx+1:]

        # Create numpy arrays with shape (nbrSequences, seqLength, dim)
        train_data = np.zeros((len(train_list), self.args['sequence_length'], len(df.columns)))
        test_data = np.zeros((len(test_list), self.args['sequence_length'], len(df.columns)))
        y_train = np.zeros((len(train_list)))
        y_test = np.zeros((len(test_list)))
        
        for i, indices in enumerate(train_list):
            train_data[i,:,:] = df.iloc[indices[:-1]].values
            y_train[i] = df.iloc[indices[-1]][self.args['target']]
        for i, indices in enumerate(test_list):
            test_data[i,:,:] = df.iloc[indices[:-1]].values
            y_test[i] = df.iloc[indices[-1]][self.args['target']]
        return train_data, test_data, y_train, y_test
        