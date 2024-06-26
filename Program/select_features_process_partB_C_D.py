#%% B部分
'''
使用A部分中high correlation filter以及select K best後的100特徵
此部分為Six Feature Order Method中三種數學原理排序方法，
標準1: 特徵間標準差與類內標準差之和的比率
標準2: 特徵間的標準差
標準3: 類內標準差之和的倒數
'''
import pandas as pd
import numpy as np
# 加載數據
labels = pd.read_csv('ML_Train.csv') 
features = pd.read_csv('training_features_total357.csv') 

#為A部分中high correlation filter以及select K best後的100個特徵
features = features.loc[:,["dTT'_y_4", 'pnn50_0', 'dPT_y_5', "dRT'_y_10", "dL'P'_x_1", "dRL'_x_9", 'pnn20_1', "dL'P'_x_0", 'bpm_4', "dRL'_x_1", "dRT'_y_11", 'ibi_0', 'dRQ_x_11', 'dRQ_x_1', 'bpm_9', 'Power_Ratio_9', "dRS'_x_4", "dST'_y_9", 'Harmonic_Power_Ratio_9', 'Low_Frequency_Power_5', 'bpm_11', 'High_Frequency_Power_1', "dST'_y_5", 'flourish_9', "dST'_y_1", "dS'T'_x_5", 'Low_Frequency_Power_9', 'pnn50_11', 'ibi_4', 'Low_Frequency_Power_10', "dRS'_x_5", 'bpm_10', 'pnn50_4', 'Low_Frequency_Power_11', 'dRS_x_10', "dL'P'_x_9", 'ibi_9', 'pnn50_9', 'dRS_x_5', 'Power_Ratio_10', 'rmssd_10', "dRT'_y_1", 'pnn50_10', 'rmssd_11', 'rmssd_0', 'ibi_11', 'flourish_5', "dL'P'_x_10", 'rmssd_9', 'ibi_10', 'pnn20_11', "dS'T'_y_0", "dL'P'_x_11", 'rmssd_4', "dRT'_y_4", "dST'_y_0", "dST'_y_10", 'Spectral_Entropy_5', 'pnn20_9', 'pnn20_10', 'dRS_x_1', 'dST_y_0', 'Power_Ratio_11', "dTT'_y_5", 'Peak_Frequency_9', 'Power_Ratio_1', 'dRS_x_0', 'Low_Frequency_Power_4', 'Power_Ratio_5', 'Spectral_Entropy_4', 'dST_y_10', 'dST_y_1', "dS'T'_x_1", 'dRS_x_11', "dST'_y_11", 'Spectral_Entropy_10', "dS'T'_x_9", 'flourish_4', 'dPT_y_1', 'Spectral_Entropy_9', 'Spectral_Entropy_11', 'Power_Ratio_0', "dRS'_x_9", 'Power_Ratio_4', "dS'T'_x_0", 'Spectral_Entropy_1', 'Spectral_Entropy_0', 'dST_y_11', 'dPT_y_0', "dS'T'_x_11", "dTT'_y_9", "dS'T'_x_10", "dRS'_x_1", "dRS'_x_10", "dTT'_y_0", "dRS'_x_0", "dRS'_x_11", "dTT'_y_1", "dTT'_y_10", "dTT'_y_11"]]

# 將特徵和標籤數據合併為一個DataFrame，並移除不需要的列
combined = pd.concat([features, labels], axis=1).drop(['SubjectId', 'age', 'sex', 'weight', 'height'], axis=1)
print(combined.columns)

# 分組計算各類別的標準差，求和
grouped_std_sum = combined.groupby('Label').std().sum()

# 計算各類別的均值，然後計算特徵間的標準差
grouped_mean = combined.groupby('Label').mean()
features_std_among_labels = grouped_mean.T.std(axis=1)

# 計算評估特徵的三個標準（criteria）
criteria1 = (features_std_among_labels / grouped_std_sum).tolist()  # 標準1: 特徵間標準差與類內標準差之和的比率
criteria2 = features_std_among_labels.to_list()  # 標準2: 特徵間的標準差
criteria3 = (1/grouped_std_sum).to_list()  # 標準3: 類內標準差之和的倒數

# 將計算的標準與特徵名稱組合成一個新的DataFrame，以便排序和篩選
criteria_df = pd.DataFrame({
    'Feature': combined.columns.drop('Label'),
    'Criteria1': criteria1,
    'Criteria2': criteria2,
    'Criteria3': criteria3
})

# 根據三種標準對特徵進行排序
sorted_criteria_df1 = criteria_df.sort_values(by='Criteria1', ascending=False)
sorted_criteria_df2 = criteria_df.sort_values(by='Criteria2', ascending=False)
sorted_criteria_df3 = criteria_df.sort_values(by='Criteria3', ascending=False)

# 選取前60個重要的特徵
top_features1 = sorted_criteria_df1.head(n=60)['Feature'].to_list()
top_features2 = sorted_criteria_df2.head(n=60)['Feature'].to_list()
top_features3 = sorted_criteria_df3.head(n=60)['Feature'].to_list()

print(top_features1)
print(top_features2)
print(top_features3)
#%% C部分
'''
此部分為將A部分、B部分中六種排序方法結合，算出每個特徵的標號平均做重新排列，
並統計每個特徵在這六種方法的出現次數，做成名為sorted_features_with_counts的列表。
'''
#隨機森林排序方法: n_estimators = 100, max_depth = 50，取前75個
top75 = [
    "dTT'_y_1", "dTT'_y_10", "dTT'_y_11", "dTT'_y_0", "Power_Ratio_0",
    "dTT'_y_9", "dRL'_x_1", "dST'_y_11", "dRQ_x_1", "dST'_y_1", "dRS'_x_0",
    "dST'_y_10", "dRS'_x_11", "Power_Ratio_4", "Low_Frequency_Power_10",
    "Low_Frequency_Power_9", "flourish_4", "dPT_y_0", "dST'_y_0",
    "dRS'_x_1", "dRS'_x_10", "Low_Frequency_Power_4", "dPT_y_1",
    "Power_Ratio_9", "dST_y_11", "Spectral_Entropy_1", "dRT'_y_1",
    "Spectral_Entropy_9", "dST_y_1", "dS'T'_x_1", "Low_Frequency_Power_11",
    "Power_Ratio_11", "dST'_y_5", "dRS'_x_9", "dRS_x_1", "dRS_x_5",
    "Power_Ratio_1", "dST_y_0", "dST_y_10", "flourish_9", "dS'T'_x_0",
    "dRQ_x_11", "Spectral_Entropy_0", "dS'T'_x_11", "dRT'_y_10",
    "dRT'_y_11", "Power_Ratio_5", "Spectral_Entropy_11", "dRS_x_10",
    "dS'T'_x_5", "dTT'_y_5", "dRS_x_11", "dS'T'_x_10", "dL'P'_x_0",
    "dS'T'_x_9", "Power_Ratio_10", "dL'P'_x_1", "dRT'_y_4", "dST'_y_9",
    "dRL'_x_9", "pnn50_9", "dL'P'_x_11", "pnn20_9", "Spectral_Entropy_5",
    "Peak_Frequency_9", "dRS'_x_4", "pnn20_10", "Spectral_Entropy_4",
    "dRS'_x_5", "High_Frequency_Power_1", "rmssd_4", "rmssd_9",
    "dL'P'_x_10", "pnn50_11", "flourish_5"
]

#隨機森林排序方法: n_estimators = 50, max_depth = 50，取前60個
top60 =  [
    "dTT'_y_1", "dTT'_y_10", "dTT'_y_11", "dTT'_y_9", "dRQ_x_1",
    "Power_Ratio_0", "dST'_y_11", "dTT'_y_0", "dRL'_x_1", "dST'_y_1",
    "dRS'_x_11", "dRS'_x_0", "dST'_y_10", "Power_Ratio_4", "dRS'_x_1",
    "dST'_y_0", "Low_Frequency_Power_9", "Low_Frequency_Power_10",
    "Low_Frequency_Power_4", "dPT_y_0", "flourish_4", "Power_Ratio_9",
    "Spectral_Entropy_1", "dPT_y_1", "dRS'_x_10", "Power_Ratio_11",
    "Spectral_Entropy_9", "dRT'_y_1", "Low_Frequency_Power_11", "dRS_x_1",
    "dS'T'_x_1", "dST_y_11", "dST'_y_5", "dST_y_1", "Power_Ratio_1",
    "dRS_x_5", "dRS'_x_9", "Spectral_Entropy_0", "dST_y_0", "dS'T'_x_10",
    "dRQ_x_11", "dST_y_10", "dS'T'_x_11", "flourish_9", "dS'T'_x_0",
    "dRS_x_10", "dL'P'_x_1", "Spectral_Entropy_11", "dRT'_y_10", "dRS_x_11",
    "dS'T'_x_9", "Power_Ratio_5", "dRT'_y_11", "dL'P'_x_0", "dS'T'_x_5",
    "dRL'_x_9", "dST'_y_9", "Power_Ratio_10", "dRT'_y_4", "dTT'_y_5"
]

#隨機森林排序方法: n_estimators = 30, max_depth = 20，取前50個
top50 = [
    "dTT'_y_1", "dTT'_y_10", "dTT'_y_11", "dTT'_y_0", "Power_Ratio_0",
    "dTT'_y_9", "dRL'_x_1", "dST'_y_11", "dRQ_x_1", "dST'_y_1", "dRS'_x_0",
    "dST'_y_10", "dRS'_x_11", "Power_Ratio_4", "Low_Frequency_Power_10",
    "Low_Frequency_Power_9", "flourish_4", "dPT_y_0", "dST'_y_0",
    "dRS'_x_1", "dRS'_x_10", "Low_Frequency_Power_4", "dPT_y_1",
    "Power_Ratio_9", "dST_y_11", "Spectral_Entropy_1", "dRT'_y_1",
    "Spectral_Entropy_9", "dST_y_1", "dS'T'_x_1", "Low_Frequency_Power_11",
    "Power_Ratio_11", "dST'_y_5", "dRS'_x_9", "dRS_x_1", "dRS_x_5",
    "Power_Ratio_1", "dST_y_0", "dST_y_10", "flourish_9", "dS'T'_x_0",
    "dRQ_x_11", "Spectral_Entropy_0", "dS'T'_x_11", "dRT'_y_10",
    "dRT'_y_11", "Power_Ratio_5", "Spectral_Entropy_11", "dRS_x_10",
    "dS'T'_x_5"]

#數學原理排序方法: 標準1的前60個特徵
list1 = ["dTT'_y_11", "dTT'_y_10", "dTT'_y_1", "dTT'_y_0", 'dST_y_11', 'Power_Ratio_0', 'Spectral_Entropy_0', 'dPT_y_0', "dRS'_x_11", "dRS'_x_0", "dTT'_y_9", 'Power_Ratio_11', 'Power_Ratio_4', 'Power_Ratio_1', "dST'_y_11", 'dPT_y_1', 'Spectral_Entropy_1', 'dST_y_10', "dRS'_x_10", "dRS'_x_1", 'Power_Ratio_5', 'Spectral_Entropy_9', 'Spectral_Entropy_11', "dS'T'_y_0", 'Spectral_Entropy_4', "dS'T'_x_10", 'dST_y_1', 'Spectral_Entropy_10', "dS'T'_x_11", 'flourish_4', "dRS'_x_9", 'Power_Ratio_10', "dTT'_y_5", "dST'_y_0", 'dST_y_0', "dS'T'_x_0", 'pnn20_10', "dS'T'_x_9", "dST'_y_10", 'pnn20_9', 'Peak_Frequency_9', "dRT'_y_1", 'Spectral_Entropy_5', 'Power_Ratio_9', "dS'T'_x_1", 'ibi_10', 'pnn20_11', 'flourish_5', 'Low_Frequency_Power_4', 'ibi_11', 'dRS_x_11', "dST'_y_1", 'pnn50_10', 'High_Frequency_Power_1', 'rmssd_4', 'bpm_10', 'Low_Frequency_Power_5', 'ibi_9', 'pnn50_9', "dRT'_y_11"]

#數學原理排序方法: 標準2的前60個特徵
list2 = ['flourish_9', 'flourish_4', 'flourish_5', 'bpm_10', 'bpm_11', 'bpm_4', 'bpm_9', "dRS'_x_0", "dRS'_x_11", "dRS'_x_10", "dRS'_x_1", "dS'T'_x_10", 'Peak_Frequency_9', "dS'T'_x_11", "dRS'_x_9", "dS'T'_x_9", "dS'T'_x_0", "dS'T'_x_1", "dRS'_x_5", "dRS'_x_4", "dS'T'_x_5", 'dRS_x_5', 'dRS_x_11', 'dRS_x_0', 'rmssd_4', 'ibi_11', 'ibi_10', 'rmssd_9', 'rmssd_0', 'rmssd_11', 'rmssd_10', 'dRS_x_1', 'ibi_9', 'ibi_0', 'dRQ_x_1', "dL'P'_x_11", "dRL'_x_9", "dRL'_x_1", "dL'P'_x_10", 'Power_Ratio_11', 'ibi_4', 'Power_Ratio_1', "dL'P'_x_9", 'Power_Ratio_9', 'Power_Ratio_5', 'Power_Ratio_0', 'Power_Ratio_4', "dL'P'_x_1", "dL'P'_x_0", 'dRS_x_10', 'Power_Ratio_10', 'dRQ_x_11', 'Spectral_Entropy_1', 'Spectral_Entropy_0', 'Spectral_Entropy_4', "dRT'_y_10", 'Spectral_Entropy_9', 'dST_y_10', 'Spectral_Entropy_5', 'Spectral_Entropy_11']

#數學原理排序方法: 標準3的前60個特徵
list3 = ['High_Frequency_Power_1', 'Low_Frequency_Power_5', 'Low_Frequency_Power_4', "dS'T'_y_0", 'Low_Frequency_Power_11', 'dPT_y_5', 'Low_Frequency_Power_10', "dTT'_y_5", "dTT'_y_4", 'Low_Frequency_Power_9', 'dPT_y_1', 'dPT_y_0', "dTT'_y_1", "dTT'_y_0", "dST'_y_0", 'pnn20_10', 'pnn20_11', 'pnn20_1', "dTT'_y_11", 'Harmonic_Power_Ratio_9', 'pnn20_9', 'pnn50_10', "dST'_y_1", 'pnn50_11', 'pnn50_9', 'pnn50_0', 'pnn50_4', "dST'_y_11", "dTT'_y_10", 'dST_y_0', "dST'_y_5", 'dST_y_1', "dTT'_y_9", 'dST_y_11', "dST'_y_10", 'Spectral_Entropy_10', 'Spectral_Entropy_11', 'Spectral_Entropy_0', 'dST_y_10', 'Spectral_Entropy_9', "dRT'_y_1", 'Spectral_Entropy_1', "dST'_y_9", 'Spectral_Entropy_4', "dRT'_y_4", 'Spectral_Entropy_5', "dRT'_y_11", "dRT'_y_10", 'Power_Ratio_0', 'Power_Ratio_4', 'Power_Ratio_10', 'dRQ_x_11', 'Power_Ratio_5', 'Power_Ratio_1', 'Power_Ratio_11', 'Power_Ratio_9', "dL'P'_x_0", 'dRS_x_10', 'ibi_4', "dL'P'_x_1"]


sets = [set(top75), set(top60), set(top50), set(list1), set(list2), set(list3)]

# 計算所有集合的交集
union = set.union(*sets)

# 轉換回列表
union_list_tot = list(union)
print(union_list_tot)

feature_indices_avg = {}
feature_counts = {}
for feature in union_list_tot:
    indices = []
    count = 0  # 初始化計數器
    if feature in list1:
        indices.append(list1.index(feature) + 1)
        count += 1  # 如果特徵在list1中出現，計數器加1
    if feature in list2:
        indices.append(list2.index(feature) + 1)
        count += 1  # 如果特徵在list2中出現，計數器加1
    if feature in list3:
        indices.append(list3.index(feature) + 1)
        count += 1  # 如果特徵在list3中出現，計數器加1
    if feature in top60:
        indices.append(top60.index(feature) + 1)
        count += 1  # 如果特徵在top60中出現，計數器加1
    if feature in top50:
        indices.append(top50.index(feature) + 1)
        count += 1  # 如果特徵在top50中出現，計數器加1
    if feature in top75:
        indices.append(top75.index(feature) + 1)
        count += 1  # 如果特徵在top75中出現，計數器加1

    # 計算平均值，避免除以零
    avg_index = sum(indices) / len(indices) if indices else 0
    feature_indices_avg[feature] = (avg_index, count)  # 將平均值和計數作為元組存儲

# 按照平均值排序字典
sorted_feature_indices = sorted(feature_indices_avg.items(), key=lambda x: x[1][0])  # 依據平均索引排序

# 提取排序好的特徵名稱並組成一個新列表，同時包括出現次數
sorted_features_with_counts = [(feature, avg_index, count) for feature, (avg_index, count) in sorted_feature_indices]

# 提取只包含特徵名稱的列表，注意修正解包方式
sorted_features_only = [feature for feature, _ in sorted_feature_indices]

# 打印排序後的特徵列表和包含次數信息
print("Sorted Features by Average Index with Occurrence Counts:")
for feature, avg_index, count in sorted_features_with_counts:
    print(f"{feature}: Avg Index - {avg_index}, Count - {count}")



#%% D部分
'''
此部分為select Feature Principle，使用下列方法進一步選出適合的特徵:
1. 平均編號的排序
2. 特徵出現次數
3. 散佈圖
4. lead 編號
5. trial and error (實際放入分類器)
'''

#1. 平均編號的排序、2. 特徵出現次數 為參考C部分的sorted_features_with_counts列表

#3.散佈圖
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature(df, feature):
    plt.figure(figsize = (50, 50), dpi=200)  
    plt.scatter(list(range(0, 12209)), df.loc[:, feature])
    plt.xlabel('Data', fontsize = 75)
    plt.ylabel(feature, fontsize = 75)
    plt.title(f'{feature} vs. Data', fontsize = 75)

featuredata = pd.read_csv("training_features_total357.csv")

plot_feature(featuredata, "dST'_y_5") #可輸入想要的特徵名稱

#4. lead 編號: 取用的7個Lead中，Lead5、9抓不到的點較多，因此在考慮特徵重要度時，這兩個Lead放在較後面。

#5. trial and error: 實際放入貝氏分類器、SVM分類器、隨機森林分類器做測試

'''
最終挑出的前50個重要特徵:
["dTT'_y_10", 'Power_Ratio_0', 'dRQ_x_1', "dTT'_y_11", "dTT'_y_1", "dST'_y_11", "dRT'_y_10", 'Spectral_Entropy_0', 'Low_Frequency_Power_5', "dRS'_x_0", "dRT'_y_1", 'Power_Ratio_9', "dRS'_x_1", "dST'_y_0", 'dRS_x_11', 'Low_Frequency_Power_11', "dTT'_y_9", "dTT'_y_0", "dRS'_x_9", "dRS'_x_11", 'Power_Ratio_1', 'Spectral_Entropy_9', "dST'_y_1", 'Power_Ratio_5', 'pnn20_10', 'Power_Ratio_4', 'Low_Frequency_Power_10', 'Low_Frequency_Power_9', 'dST_y_11', 'flourish_4', "dRL'_x_1", 'dST_y_0', "dRT'_y_11", "dS'T'_x_0", 'High_Frequency_Power_1', 'Spectral_Entropy_1', "dST'_y_10", 'flourish_9', 'dRS_x_5', 'Spectral_Entropy_5', 'dRS_x_10', 'dRQ_x_11', 'Low_Frequency_Power_4', 'dPT_y_0', "dST'_y_9", 'pnn20_9', 'pnn50_9', 'pnn20_11', "dL'P'_x_0", "dL'P'_x_11"]

並分別以不同數量帶入貝氏分類器、SVM分類器、隨機森林分類器，
發現貝氏分類器取前35個特徵可達最高準確率；
發現SVM分類器取前40個特徵可達最高準確率；
發現隨機森林分類器取前45個特徵可達最高準確率；
'''

