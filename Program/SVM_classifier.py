import numpy as np
import prepare_train_test as ptt  # 導入自定義模塊，用於數據的預處理

# 使用從select_features_process.py選出的前40個特徵，這些特徵達到了最高準確度
select_feature = ["dTT'_y_10", 'Power_Ratio_0', 'dRQ_x_1', "dTT'_y_11", "dTT'_y_1", "dST'_y_11", "dRT'_y_10", 'Spectral_Entropy_0', 'Low_Frequency_Power_5', "dRS'_x_0", "dRT'_y_1", 'Power_Ratio_9', "dRS'_x_1", "dST'_y_0", 'dRS_x_11', 'Low_Frequency_Power_11', "dTT'_y_9", "dTT'_y_0", "dRS'_x_9", "dRS'_x_11", 'Power_Ratio_1', 'Spectral_Entropy_9', "dST'_y_1", 'Power_Ratio_5', 'pnn20_10', 'Power_Ratio_4', 'Low_Frequency_Power_10', 'Low_Frequency_Power_9', 'dST_y_11', 'flourish_4', "dRL'_x_1", 'dST_y_0', "dRT'_y_11", "dS'T'_x_0", 'High_Frequency_Power_1', 'Spectral_Entropy_1', "dST'_y_10", 'flourish_9', 'dRS_x_5', 'Spectral_Entropy_5']


pred_file_path = './SVM.csv'  # 存放測試數據集預測結果的文件路徑

# 讀取和預處理訓練數據
train = ptt.Training(select_feature)
X_train, Y_train, X_valid, Y_valid = train.feature_label()

# 讀取和預處理測試數據
test = ptt.Testing(select_feature)
testset = test.read_data()
X_test = testset.loc[:, select_feature]
X_test = np.array(X_test.values, dtype=float)

# Standard Scaler函數，用於特徵標準化
def standard_scaler(X):
    mean = np.mean(X, axis=0)  # 計算每個特徵的平均值
    std = np.std(X, axis=0)    # 計算每個特徵的標準差
    return (X - mean) / std    # 返回標準化的特徵數據

X_train_std = standard_scaler(X_train)  # 對訓練數據進行標準化
X_test_std = standard_scaler(X_test)    # 對測試數據進行標準化

# 簡單的SVM類實現
class SimpleSVM:
    def __init__(self):
        self.weights = None  # 初始化權重為None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # 初始化權重為零
        for _ in range(200):  # 進行200次訓練
            for idx, x_i in enumerate(X):
                if np.dot(x_i, self.weights) < 1:  # SVM的條件判斷
                    # 更新權重 (0.0000001為學習率；0.02為正規化參數)
                    self.weights += 0.0000001 * ((y[idx] * x_i) - (0.02 * self.weights))
            print(_/1000*100,"%")

    def predict(self, X):
        # 使用學到的權重對新數據進行預測
        return np.dot(X, self.weights)

# 多類別SVM實現，使用一對其餘（One-vs-Rest）策略
class MultiClassSVM:
    def __init__(self, n_classes):
        self.classifiers = [SimpleSVM() for _ in range(n_classes)]  # 為每個類別創建一個SVM分類器
        self.n_classes = n_classes

    def fit(self, X, y):
        # 訓練每個分類器對應一個類別
        for i in range(self.n_classes):
            binary_labels = np.where(y == i, 1, -1)  # 創建二元標籤
            self.classifiers[i].fit(X, binary_labels)

    def predict(self, X):
        # 從所有分類器中獲得預測結果，選擇決策函數值最高的類別
        predictions = np.array([clf.predict(X) for clf in self.classifiers]).T
        return np.argmax(predictions, axis=1)

# 使用已定義和預處理好的X_train_std, y_train, X_test_std進行模型訓練
svm = MultiClassSVM(n_classes=4)
svm.fit(X_train_std, Y_train)

# 訓練和驗證模型
y_val_pred = svm.predict(X_valid)
val_accuracy = np.mean(y_val_pred == Y_valid)
print(f"Validation Accuracy: {val_accuracy}")

# 預測測試數據
y_test_pred = svm.predict(X_test_std)

# 保存預測結果
testset['pred'] = y_test_pred
data_test_result = testset[['SubjectId', 'pred']].copy()
data_test_result['Label'] = data_test_result['pred']
data_test_result.drop(columns='pred', inplace=True)
data_test_result.to_csv(pred_file_path, index=False)

print("Predictions saved to", pred_file_path)

