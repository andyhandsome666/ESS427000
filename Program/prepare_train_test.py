import numpy as np
import pandas as pd

#%%
'''
def read_data(feature_file_path, label_file_path, select_feature, option=True): # 讀feature並加上label
    data_feature = pd.read_csv(feature_file_path) # 從 training/testing_feature 讀 feature
    data_label   = pd.read_csv(label_file_path)   # 從 ML_train/test 讀 label
    data_feature = data_feature[select_feature]   # 從 input 選 select feature
    
    data_feature = data_feature.reset_index(drop=True) # 修改排序避免錯誤
    data_label   = data_label.reset_index(drop=True)
    
    data = pd.DataFrame()
    data['SubjectId'] = data_label['SubjectId']
    if option:                                    # 當模式為 train 需要轉換 label 為數字
        data['Label'] = data_label['Label'].replace({'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3})
    data['pred'] = None                           # 設定空 prediction
    data = pd.concat([data, data_feature], axis=1)  # 將 subject ID、label、prediction 接在一起
                
    return data

def expand_data(data): # 增加疾病的數據
    data_norm = data[data['Label'] == 0].copy()
    new_data = data_norm.copy()
    
    for label in range(1,4):
        class_data = data[data['Label'] == label].copy()
        new_class_data = class_data.copy()
        add_dict = {1: range(4), 2: range(2), 3: range(3)} # 不同label增加不同倍
        add = add_dict.get(label)
        
        for i in add: #加上~倍的數據
            noise = np.random.normal(0, 0.005, size=(class_data.shape[0], class_data.shape[1]-3))
            noisy_data = class_data.copy()
            noisy_data.iloc[:, 3:] += noise
            new_class_data = pd.concat([new_class_data, noisy_data], axis=0)
            new_class_data.reset_index(drop=True, inplace=True)
    
        new_data = pd.concat([new_data, new_class_data], axis=0)
        new_data.reset_index(drop=True, inplace=True)
    return new_data

def split_train_test(data): # 分train/test dataset
    data = data.reset_index(drop=True)          # 修改排序避免錯誤
    np.random.seed(40)                          # 產生 random seed
    rand_seq = np.random.permutation(len(data)) # 產生隨機序列
    
    val_size = round(len(data) * 0.2)           # train / test = 8:2
    val_index = rand_seq[:val_size]             # index 取用 random sequence
    
    data_validation = data.iloc[val_index]      # 選取 valid index
    data_train      = data.drop(val_index)      # 扣掉 valid index
    
    data_validation = data_validation.reset_index(drop=True) # 修改排序避免錯誤
    data_train      = data_train.reset_index(drop=True)
    return data_train, data_validation
'''
#%%
class DATA_format:
    def __init__(self, select_feature):
        self._feature_file_path = ''
        self._label_file_path   = ''
        self._select_feature    = select_feature
        self._option            = True
        
    def read_data(self):
        data_feature = pd.read_csv(self._feature_file_path) # 從 training/testing_feature 讀 feature
        data_label   = pd.read_csv(self._label_file_path)   # 從 ML_train/test 讀 label
        data_feature = data_feature[self._select_feature]   # 從 input 選 select feature
        
        data_feature = data_feature.reset_index(drop=True)  # 修改排序避免錯誤
        data_label   = data_label.reset_index(drop=True)
        
        data = pd.DataFrame()
        data['SubjectId'] = data_label['SubjectId']
        if self._option:                                    # 當模式為 train 需要轉換 label 為數字
            data['Label'] = data_label['Label'].replace({'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3})
        data['pred'] = None                                 # 設定空 prediction
        data = pd.concat([data, data_feature], axis=1)      # 將 subject ID、label、prediction 接在一起
        
        print('='*30); print('Finish reading data.')
        return data

class Training(DATA_format):
    def __init__(self, select_feature):
        super().__init__(select_feature)
        self._feature_file_path = './training_features_total406.csv'#'./training_features_1000_50.csv'#'./train_0531.csv'#
        self._label_file_path   = './ML_Train.csv'
        
    def _expand_data(self): # 增加疾病的數據
        data = self.read_data()
        data_norm = data[data['Label'] == 0].copy()
        new_data  = data_norm.copy()
        
        for label in range(1,4):
            class_data = data[data['Label'] == label].copy()
            new_class_data = class_data.copy()
            add_dict = {1: range(4), 2: range(2), 3: range(3)} # 不同label增加不同倍
            add = add_dict.get(label)
            
            for _ in add: #加上~倍的數據
                noise = np.random.normal(0, 0.005, size=(class_data.shape[0], class_data.shape[1]-3))
                noisy_data = class_data.copy()
                noisy_data.iloc[:, 3:] += noise
                new_class_data = pd.concat([new_class_data, noisy_data], axis=0)
                new_class_data.reset_index(drop=True, inplace=True)
        
            new_data = pd.concat([new_data, new_class_data], axis=0)
            new_data.reset_index(drop=True, inplace=True)
            
        print('='*30); print('Finish expanding data.')
        return new_data
    
    def split_train_test(self, random_num=20): # 分train/test dataset
        data = self._expand_data()
        data['pred'] = None    
        data = data.reset_index(drop=True)          # 修改排序避免錯誤
        np.random.seed(random_num)                  # 產生 random seed
        rand_seq = np.random.permutation(len(data)) # 產生隨機序列
        
        val_size  = round(len(data) * 0.2)          # train / test = 8:2
        val_index = rand_seq[:val_size]             # index 取用 random sequence
        
        data_validation = data.iloc[val_index]      # 選取 valid index
        data_train      = data.drop(val_index)      # 扣掉 valid index
        
        data_validation = data_validation.reset_index(drop=True) # 修改排序避免錯誤
        data_train      = data_train.reset_index(drop=True)
        
        print('='*30); print('Finish splitting data.')
        return data_train, data_validation
    
    def feature_label(self):
        data_train, data_validation = self.split_train_test()
        X_train = np.array(data_train.loc[:, self._select_feature].values, dtype=float)
        Y_train = data_train.loc[:, 'Label'].values
        X_valid = np.array(data_validation.loc[:, self._select_feature].values, dtype=float)
        Y_valid = data_validation.loc[:, 'Label'].values
        return X_train, Y_train, X_valid, Y_valid
        
class Testing(DATA_format):
    def __init__(self, select_feature):
        super().__init__(select_feature)
        self._feature_file_path = './testing_features_total406.csv'#'./testing_features_1000_50.csv'#'./testing_0531.csv' #
        self._label_file_path   = './ML_Test.csv' 
        self._option            = False

        
        
    