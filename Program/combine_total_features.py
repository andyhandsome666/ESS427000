#%%將train data所有特徵合併: 7個lead，34個ecg特徵、9個spectrum特徵;8個heartbeat特徵;共(34+9+8)*7=357個特徵
import pandas as pd
# 從ecg_features.py中生成的34個ecg特徵匯入
train_data = pd.read_csv('training_features_7lead.csv')

# 從other_features.py中生成的9個spectrum特徵匯入
train_data_frequency = pd.read_csv('spectrum_training_features.csv')

# 從other_features.py中生成的7個heartbeat特徵匯入
train_data_heartbeat = pd.read_csv('heartbeat_training_features.csv')

# 將全部合併
train_data_total = pd.concat([train_data, train_data_frequency, train_data_heartbeat], axis=1)
# 存成最終的所有特徵csv檔
train_data_total.to_csv("training_features_total357.csv", index=False)


#%%將test data所有特徵合併: 7個lead，34個ecg特徵、9個spectrum特徵;8個heartbeat特徵;共(34+9+8)*7=357個特徵
import pandas as pd
# 從ecg_features.py中生成的34個ecg特徵匯入
test_data = pd.read_csv('testing_features_7lead.csv')

# 從other_features.py中生成的9個spectrum特徵匯入
test_data_frequency = pd.read_csv('spectrum_testing_features.csv')

# 從other_features.py中生成的7個heartbeat特徵匯入
test_data_heartbeat = pd.read_csv('heartbeat_testing_features.csv')

# 將全部合併
test_data_total = pd.concat([test_data, test_data_frequency, test_data_heartbeat], axis=1)
# 存成最終的所有特徵csv檔
test_data_total.to_csv("testing_features_total357.csv", index=False)

