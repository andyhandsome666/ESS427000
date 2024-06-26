#%% A部分
'''
此部分包含: 第一步: high correlation filter (threshold > 0.9)、
           第二步: select K best 選前100個重要特徵
           第三步: Six Feature Order Method中的隨機森林選取特徵排序
           (參數分別為: n_estimators = 30,  max_depth = 20，取前50個；
                       n_estimators = 50,  max_depth = 50，取前60個；
                       n_estimators = 100, max_depth = 50，取前75個)
'''
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

# 加載目標變量
def load_data():
    """
    加載訓練數據和特徵數據，並進行相關性特徵選擇

    返回:
    - X_filtered: 經過相關性過濾後的特徵數據
    - y: 目標變量數據
    - filtered_feature_names: 過濾後的特徵名稱
    """
    # 加載目標變量
    target_df = pd.read_csv('ML_Train.csv')
    y = target_df['Label'].replace({'NORM': 0, 'STTC': 1, 'CD': 2, 'MI': 3}).values

    # 加載特徵數據
    features_df = pd.read_csv('training_features_total357.csv')
    # 處理缺失值，使用列均值填充
    X = np.nan_to_num(features_df.values, nan=np.nanmean(features_df.values, axis=0))

    # 第一步: high correlation filter (threshold > 0.9)
    corr_matrix = np.corrcoef(X, rowvar=False)
    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    to_drop = [features_df.columns[i] for i in range(len(features_df.columns)) if any(corr_matrix[i, upper[i]] > 0.90)]
    X_filtered = features_df.drop(columns=to_drop).values
    filtered_feature_names = [feature for feature in features_df.columns if feature not in to_drop]

    return X_filtered, y, filtered_feature_names

# 第二步: select K best 選前100個重要特徵
def f_classif(X, y):
    """
    使用 ANOVA F 測試進行特徵選擇

    參數:
    - X: 特徵數據
    - y: 目標變量

    返回:
    - F: 每個特徵的 F 值
    """
    F = []
    for i in range(X.shape[1]):
        classes = np.unique(y)  # 獲取所有的類別
        mean_total = np.mean(X[:, i])  # 計算特徵的總均值
        sst = np.sum((X[:, i] - mean_total) ** 2)  # 計算總平方和
        ssw = 0  # 組內平方和
        ssb = 0  # 組間平方和
        for c in classes:
            X_c = X[y == c, i]
            mean_c = np.mean(X_c)
            ssb += len(X_c) * (mean_c - mean_total) ** 2
            ssw += np.sum((X_c - mean_c) ** 2)
        F.append((ssb / (len(classes) - 1)) / (ssw / (len(X) - len(classes))))
    return np.array(F)

# 第三步: Six Feature Order Method中的隨機森林選取特徵排序
class SimpleDecisionTree:
    def __init__(self, max_depth=5, random_state=None):
        """
        初始化決策樹參數

        參數:
        - max_depth: 決策樹最大深度
        - random_state: 隨機種子
        """
        self.max_depth = max_depth
        self.random_state = random_state
        self.tree_ = None
        self.feature_importances_ = np.zeros(0)

    def fit(self, X, y):
        """
        擬合決策樹模型

        參數:
        - X: 特徵數據
        - y: 目標變量
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.tree_ = self._grow_tree(X, y)
        self.feature_importances_ = np.zeros(X.shape[1])
        self._compute_feature_importance(self.tree_)

    def _grow_tree(self, X, y, depth=0):
        """
        遞歸構建決策樹

        參數:
        - X: 特徵數據
        - y: 目標變量
        - depth: 當前深度

        返回:
        - node: 節點
        """
        num_samples = y.size  # 當前節點的樣本數量
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]  # 每個類別的樣本數量
        if len(num_samples_per_class) == 0:
            return None
        predicted_class = np.argmax(num_samples_per_class)  # 預測類別為樣本數最多的類別

        node = {'type': 'leaf', 'class': predicted_class, 'num_samples': num_samples}
        if depth < self.max_depth and num_samples > 2:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                if len(y_left) == 0 or len(y_right) == 0:
                    return node
                node = {'type': 'node', 'index': idx, 'threshold': thr, 'num_samples': num_samples,
                        'left': self._grow_tree(X_left, y_left, depth + 1),
                        'right': self._grow_tree(X_right, y_right, depth + 1)}
        return node

    def _best_split(self, X, y):
        """
        找到最佳分裂點

        參數:
        - X: 特徵數據
        - y: 目標變量

        返回:
        - best_idx: 最佳特徵索引
        - best_thr: 最佳分裂閾值
        """
        best_idx, best_thr = None, None
        best_gain = -np.inf
        m = int(np.sqrt(X.shape[1]))  # 考慮的特徵數量
        for idx in np.random.choice(X.shape[1], m, replace=False):
            thresholds = np.unique(X[:, idx])
            for thr in thresholds:
                gain = self._gini_gain(y, X[:, idx] < thr)
                if gain > best_gain:
                    best_gain = gain
                    best_idx, best_thr = idx, thr
        return best_idx, best_thr

    def _gini(self, y):
        """
        計算基尼不純度

        參數:
        - y: 目標變量

        返回:
        - 基尼不純度
        """
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _gini_gain(self, y, split_mask):
        """
        計算基尼增益

        參數:
        - y: 目標變量
        - split_mask: 分裂掩碼

        返回:
        - 基尼增益
        """
        y_left, y_right = y[split_mask], y[~split_mask]
        m = len(y)
        return self._gini(y) - (len(y_left) / m * self._gini(y_left) + len(y_right) / m * self._gini(y_right))

    def _compute_feature_importance(self, node):
        """
        計算特徵重要性

        參數:
        - node: 當前節點
        """
        if node is not None and node['type'] == 'node':
            self.feature_importances_[node['index']] += node['num_samples']
            self._compute_feature_importance(node['left'])
            self._compute_feature_importance(node['right'])

    def predict(self, X):
        """
        預測方法占位符
        """
        pass

# 用於並行處理的訓練函數
def train_tree(args):
    """
    訓練決策樹

    參數:
    - args: 包含決策樹、特徵數據和目標變量的元組

    返回:
    - 特徵重要性
    """
    tree, X, y = args
    tree.fit(X, y)
    return tree.feature_importances_

# 使用多處理和tqdm的隨機森林實現
class SimpleRandomForestClassifier:
    def __init__(self, n_estimators=100, random_state=42, max_depth=50):
        """
        初始化隨機森林參數

        參數:
        - n_estimators: 樹的數量
        - random_state: 隨機種子
        - max_depth: 樹的最大深度
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.trees = [SimpleDecisionTree(max_depth=self.max_depth, random_state=random_state + i) for i in range(n_estimators)]

    def fit(self, X, y):
        """
        擬合隨機森林模型

        參數:
        - X: 特徵數據
        - y: 目標變量
        """
        with mp.Pool(mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap(train_tree, [(tree, X, y) for tree in self.trees]), total=self.n_estimators))
        self.feature_importances_ = np.mean(results, axis=0)

def main():
    """
    主函數，執行數據加載、特徵選擇和模型訓練
    """
    # 加載數據
    X_filtered, y, filtered_feature_names = load_data()

    # 使用 ANOVA F 測試進行特徵選擇
    scores = f_classif(X_filtered, y)
    k_best_indices = np.argsort(scores)[-100:]  # 選擇F值最高的100個特徵
    X_new = X_filtered[:, k_best_indices]
    selected_features_kbest = [filtered_feature_names[i] for i in k_best_indices]
    print("Selected features using manual F-test:", selected_features_kbest)

    # 訓練隨機森林模型
    model = SimpleRandomForestClassifier(n_estimators=100, random_state=42, max_depth=10) #n_estimators為樹的數量；max_depth為樹的深度，參數可調整
    model.fit(X_new, y)
    importances = model.feature_importances_
    feature_importances = pd.DataFrame(importances, index=selected_features_kbest, columns=['importance'])
    feature_importances.sort_values(by='importance', ascending=False, inplace=True)

    # 打印最終選擇的特徵
    print("Final selected features using manual feature importance:", feature_importances.head(60).index) #取前60個，參數可調整

if __name__ == "__main__":
    main()
