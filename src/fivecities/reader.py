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
    categorical_dummy = ['season_1', 'season_2', 'season_3', 'season_4', 'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv, cbwd_SW']
    
    default_args = {
        'sequence_length' : 12,
        'gap' : 6,
        'target' : 'PM_US Post',
        'city_list' : city_list,
        'test_portion' : 0.2,
        'scale' : False,            # mean_imputation and scale is done in the training loop ((experiment.py) instead of in this class
        'mean_imputation' : False,  # mean_imputation and scale is done in the training loop (experiment.py) instead of in this class
        'hayashi' : False,
        'classification' : False
        }
       
    hayashi_values = {
        'freq' : 1,
        'k' : 14,
        'threshold': 25
    }

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

            # Code added to replicate Hayashi et al (2019)
            if self.args['hayashi']:
                
                #self.args['sequence_length'] = self.hayashi_values['k'] # To make k=15 step-ahead prediction
            
                # Add datetime for indexing
                self.df_dict[c]['datetime'] = pd.to_datetime(self.df_dict[c][['year','month','day']])
                
                # Dummy encoding of categorical variables
                self.df_dict[c] = pd.get_dummies(self.df_dict[c], columns=FiveCities.categorical_list)

                freq = FiveCities.hayashi_values['freq'] # This is to get mean of every four days
                date_range = pd.date_range(self.df_dict[c].iloc[0]['datetime'], self.df_dict[c].iloc[-1]['datetime'], freq=f'{freq}D')

                # Loop over starting days
                tmp_list = []
                idx = 0
                while idx < len(date_range) - 1:

                    # Compute between start_date and end_date (which is a four day interval)
                    start_date = date_range[idx]
                    end_date = date_range[idx+1]
                    mask = (self.df_dict[c]['datetime'] > start_date) & (self.df_dict[c]['datetime'] <= end_date)
                    tmp_list.append(self.df_dict[c][mask].mean(numeric_only=True)) 
                    idx += 1

                # Create new df with means of every four days
                self.df_dict[c] = pd.DataFrame(tmp_list)

                # Subset with relevant features
                features = FiveCities.feature_list + [self.args['target']] + ['timestamp']
                for f in FiveCities.categorical_list:
                    features.remove(f)
                for f in FiveCities.categorical_dummy:
                    if f in self.df_dict[c].columns:
                        features.append(f)

                self.df_dict[c] = self.df_dict[c][features]  # timestamp will be removed later in filter_df method
            
            else: # Ordinary code from our work
                # Subset with relevant features
                self.df_dict[c] = self.df_dict[c][(FiveCities.feature_list + [self.args['target']]+['timestamp'])]  # timestamp will be removed later in filter_df method
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

            # Hayashi uses classification
            if self.args['hayashi']: self.args['classification'] = True
            
            # Turn the target values into binary values for classification 
            if self.args['classification']: # TODO: Check threshold value
                y_train = X_train[:,0,self.header[c].index(self.args['target'])] < y_train
                y_test  = X_test[:,0,self.header[c].index(self.args['target'])] < y_test

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

        # TODO: Add min-max scaler for Hayashi?

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
        

        if not self.args['hayashi']:
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

            if len(tmp_idx_list) >= self.args['sequence_length'] + 1:
                idx_list.append(tmp_idx_list)

        else:
            # If Hayashi replication, we have overlapping sequences to increase sample size
            # TODO: Ensure that training and test set do not overlap
            idx_list = []
            for idx in range(0, len(df)-(self.args['sequence_length'] + 1)):
                tmp_idx_list = list(range(idx, idx + self.args['sequence_length'] + 1))
                idx_list.append(tmp_idx_list)        

        df = df.drop(labels=['timestamp'], axis=1) # do not include timestamp in final features
        self.header[city] = list(self.df_dict[city].columns)
            
        # Split data into training and test set
        split_idx = int(len(idx_list)*(1-self.args['test_portion'])) 
        train_list = idx_list[:split_idx]
        if not self.args['hayashi']:
            test_list = idx_list[split_idx+1:]
        else:
            test_list = idx_list[split_idx+self.args['sequence_length']+1:] # To ensure gap between training and test set for the Hayashi replication

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
        

if __name__ == "__main__":

    fc = FiveCities("/Users/rickardkarlsson/Documents/Chalmers/MSc/Thesis/LearningUsingPrivilegedTimeSeries/data/fivecities/", args={'hayashi' : True})
    print(fc.sample('beijing')[0].shape)
    print(fc.sample('beijing')[1].shape)
    
    import matplotlib.pyplot as plt
    