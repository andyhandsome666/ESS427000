1. 執行 ecg_features.py：找心電圖特徵；other_features.py：找其他特徵

2. 執行 combine_total_feature.py：
將1. 中的所有特徵統整成 training_features_total357.csv/ testing_features_total357.csv

3. 執行 select_features_process_partA.py ：
	(1) 用 high correlation filter + select K best 篩出100個特徵
	(2) 用隨機森林排序：
		n = 30,    max = 20, 取前50個
		n = 50,    max = 50, 取前60個
		n = 100, max = 50, 取前75個

4. 執行 select_features_process_partB_C_D.py
	(1) 三種數學原理排序
	(2) 結合六種排序方法
	(3) 依以下五點找合適特徵：
		a. 平均編號
		b. 出現次數
		c. 散佈圖
		d. lead 編號
		e. trial and error

5. 執行 Random_Forest_classifier

* 我們附上其他兩種分類器，但最後上傳結果使用隨機森林，得到最終的預測結果為 Team_1.csv
* 我們隨機森林在設計的時候沒有固定random state，所以可能會生出不同的結果 
* 附上training_features_total357.csv、testing_features_total357.csv 為所有特徵的csv檔