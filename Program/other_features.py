#以下程式如果為生成 testing data，將train名稱改為test, 12209改為6000
#%% A部分 heartbeat相關特徵
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.signal as sig

measures = {}  # 用於存儲計算得到的心率指標

def get_data(filename):
    # 從CSV文件加載數據
    dataset = pd.read_csv(filename)
    return dataset

def rolmean(dataset, hrw, fs):
    # 計算心電信號的移動平均，用於平滑數據並幫助後續的峰值檢測
    # hrw (half rolling window) 表示滑動窗口的一半寬度，單位是秒
    mov_avg = dataset['hart'].rolling(int(hrw*fs)).mean()  # 應用滑動平均窗口
    avg_hr = (np.mean(dataset.hart))  # 計算全局平均心率
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]  # 處理NaN值
    mov_avg = [x*1.2 for x in mov_avg]  # 提高移動平均的閾值以避免噪聲影響
    dataset['hart_rollingmean'] = mov_avg  # 儲存結果

def detect_peaks(dataset):
    # 檢測心電信號中的R峰，這些峰值對應於心臟的每次搏動
    window = []
    peaklist = []
    listpos = 0
    for datapoint in dataset.hart:
        rollingmean = dataset.hart_rollingmean[listpos]
        if (datapoint < rollingmean) and (len(window) < 1):
            listpos += 1
        elif (datapoint > rollingmean):
            window.append(datapoint)
            listpos += 1
        else:
            maximum = max(window)
            beatposition = listpos - len(window) + (window.index(maximum))
            peaklist.append(beatposition)
            window = []
            listpos += 1
    measures['peaklist'] = peaklist
    measures['ybeat'] = [dataset.hart[x] for x in peaklist]

def calc_RR(dataset, fs):
    # 計算R-R間隔，即連續兩次心跳之間的時間間隔，通常用於分析心率變異性
    peaklist = measures['peaklist']
    RR_list = [((peaklist[i+1] - peaklist[i]) / fs) * 1000.0 for i in range(len(peaklist)-1)]
    measures['RR_list'] = RR_list  # 儲存R-R間隔列表
    RR_diff = [abs(RR_list[i] - RR_list[i+1]) for i in range(len(RR_list)-1)]
    RR_sqdiff = [x**2 for x in RR_diff]
    measures['RR_diff'] = RR_diff
    measures['RR_sqdiff'] = RR_sqdiff  # 平方差用於計算RMSSD（Root Mean Square of Successive Differences）

def calc_ts_measures():
    # 計算時域心率變異性指標
    RR_list = measures['RR_list']
    measures['bpm'] = 60000 / np.mean(RR_list)  # 平均心率，每分鐘心跳次數
    measures['ibi'] = np.mean(RR_list)  # 平均R-R間隔
    measures['sdnn'] = np.std(RR_list)  # 標準差
    measures['sdsd'] = np.std(measures['RR_diff'])  # 連續差分的標準差
    measures['rmssd'] = np.sqrt(np.mean(measures['RR_sqdiff']))  # RMSSD值
    NN20 = [x for x in measures['RR_diff'] if x > 20]  # 統計差分大於20毫秒的次數
    NN50 = [x for x in measures['RR_diff'] if x > 50]  # 統計差分大於50毫秒的次數
    measures['nn20'] = NN20
    measures['nn50'] = NN50
    measures['pnn20'] = float(len(NN20)) / float(len(measures['RR_diff']))
    measures['pnn50'] = float(len(NN50)) / float(len(measures['RR_diff']))

def plotter(dataset, title):
    # 繪製心電圖和檢測到的峰值
    plt.title(title)
    plt.plot(dataset.hart, alpha=0.5, color='blue', label="raw signal")
    plt.plot(dataset.hart_rollingmean, color='green', label="moving average")
    plt.scatter(measures['peaklist'], measures['ybeat'], color='red', label="average: %.1f BPM" % measures['bpm'])
    plt.legend(loc=4, framealpha=0.6)
    plt.show()

def process(dataset, hrw, fs):
    # 處理心電信號數據的主函數
    rolmean(dataset, hrw, fs)
    detect_peaks(dataset)
    calc_RR(dataset, fs)
    calc_ts_measures()

def denoise(data, order=4, lowcut=20, fs=500):
    # 使用低通濾波器進行去噪
    b, a = sig.butter(order, lowcut, 'lowpass', fs=fs)
    sig_denoised = sig.filtfilt(b, a, data)
    return sig_denoised
    
def Flourish(data):
    # 計算傅立葉變換的功率
    fs = 500
    fft_result = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(fft_result), 1/fs)
    power_spectrum = np.abs(fft_result) ** 2
    total_power = np.sum(power_spectrum)
    return total_power

origindata = np.load('ML_Train.npy', mmap_mode='r') 

leads = [0, 1, 4, 5, 9, 10, 11]  # 選定導程
for j in leads: #遍歷導程
    lead = j
    newfeaturedata=pd.DataFrame(data=None)
    for i in range(12209): #遍歷人數
        ecg_data = origindata[i, lead, :]
        sig_denoised = denoise(ecg_data)
        sig_denoised = pd.DataFrame(sig_denoised, columns=['hart'])
        process(sig_denoised, 0.03, 500)
        newfeaturedata.loc[i,f'bpm_{lead}'] = measures['bpm']
        newfeaturedata.loc[i,f'ibi_{lead}'] = measures['ibi']
        newfeaturedata.loc[i,f'sdnn_{lead}'] = measures['sdnn']
        newfeaturedata.loc[i,f'sdsd_{lead}'] = measures['sdsd']
        newfeaturedata.loc[i,f'rmssd_{lead}'] = measures['rmssd']
        newfeaturedata.loc[i,f'pnn20_{lead}'] = measures['pnn20']
        newfeaturedata.loc[i,f'pnn50_{lead}'] = measures['pnn50']
        newfeaturedata.loc[i, f'flourish_{lead}'] = Flourish(sig_denoised)
        print("completed:",round((i/12209)*100,3),"%")
    
    newfeaturedata.to_csv(f"heartbeat_featuredata_lead{lead}.csv", index=False)
#%% 將每個取用導程的hearbeat特徵合併
import numpy as np
import pandas as pd
featuredata0 = pd.read_csv("heartbeat_featuredata_lead0.csv")

featuredata1 = pd.read_csv("heartbeat_featuredata_lead1.csv")

featuredata4 = pd.read_csv("heartbeat_featuredata_lead4.csv")
featuredata4.drop('class', axis=1, inplace=True)
featuredata5 = pd.read_csv("heartbeat_featuredata_lead5.csv")

featuredata9 = pd.read_csv("heartbeat_featuredata_lead9.csv")

featuredata10 = pd.read_csv("heartbeat_featuredata_lead10.csv")
featuredata10.drop('class', axis=1, inplace=True)
featuredata11 = pd.read_csv("heartbeat_featuredata_lead11.csv")
featuredata11.drop('class', axis=1, inplace=True)
new_data = pd.concat([featuredata0, featuredata1, featuredata4, featuredata5, featuredata9, featuredata10, featuredata11], axis=1)

data = pd.DataFrame()
#data['SubjectId'] = np.arange(0, 6000)
#data['pred'] = None
data = pd.concat([data, new_data], axis=1)
data.to_csv("heartbeat_training_features.csv", index=False)


#%% B部分 spectrum相關特徵
import pandas as pd
import scipy.signal as sig
import numpy as np
from scipy.fftpack import fft
from scipy.signal import welch
from scipy.stats import entropy

def denoise(data, order=4, lowcut=20, fs=500):
    # 使用Butterworth低通濾波器對數據進行去噪。這種濾波器常用於減少高頻噪音，保留低頻信號成分。
    # 參數 order 指定濾波器的階數，lowcut 指定截止頻率，fs 指定數據的採樣頻率。
    b, a = sig.butter(order, lowcut, 'lowpass', fs=fs)
    sig_denoised = sig.filtfilt(b, a, data)  # 使用雙向濾波避免引入相位偏移。
    return sig_denoised
    

origindata = np.load('ML_Train.npy', mmap_mode='r')  

leads = [0, 1, 4, 5, 9, 10, 11]  # 選定導程
for j in leads: # 遍歷導程
    lead = j
    newfeaturedata = pd.DataFrame(data=None)
    for i in range(12209): # 遍歷人數
        ecg_data = origindata[i, lead, :]
        sig_denoised = denoise(ecg_data)
    
        sig_denoised = sig_denoised - np.mean(sig_denoised)  # 去除直流分量，將信號均值調整為0，有助於進一步分析。
        fs = 500  # 設定採樣頻率
        
        # FFT 變換，將時域信號轉換為頻域信號，以便分析其頻率成分。
        ecg_fft = fft(sig_denoised)
        freq = np.linspace(0.0, fs/2, int(fs/2))

        # 計算單邊頻譜，只考慮正頻率部分，因為FFT的結果是對稱的。
        ecg_fft_half = 2.0/fs * np.abs(ecg_fft[:fs//2])

        # 使用 Welch 方法計算功率譜密度，該方法可以平滑估計的功率譜，使之更便於識別信號中的主要成分。
        f, psd = welch(sig_denoised, fs, nperseg=1024)
    
        # 計算特定頻帶內的功率，並計算低頻與高頻功率比。
        power_low = np.sum(psd[(f >= 5) & (f <= 15)])
        power_high = np.sum(psd[(f >= 15) & (f <= 40)])
        power_ratio = power_low / power_high if power_high > 0 else 0
    
        # 諧波分析，查找基頻及其倍頻的能量，有助於了解信號的周期性。
        base_freq = freq[np.argmax(ecg_fft_half)]
        harmonics = base_freq * np.arange(1, 6)
        harmonic_powers = [np.sum(psd[np.isclose(f, h, atol=1.0)]) for h in harmonics]
        total_harmonic_power = np.sum(harmonic_powers)
        average_harmonic_power = np.mean(harmonic_powers)
        max_harmonic_power = np.max(harmonic_powers)
        harmonic_power_ratio = harmonic_powers[0] / total_harmonic_power if total_harmonic_power > 0 else 0
    
        # 譜熵計算，用於衡量信號頻譜的熵值，反映了頻譜的隨機性和複雜度。
        normalized_psd = psd / np.sum(psd)  # 正規化 PSD 使其總和為1，以計算熵。
        spectral_entropy = entropy(normalized_psd)
        
        # 將計算出的特徵保存到DataFrame中。
        newfeaturedata.loc[i,f'Peak_Frequency_{lead}'] = base_freq
        newfeaturedata.loc[i,f'Low_Frequency_Power_{lead}'] = power_low
        newfeaturedata.loc[i,f'High_Frequency_Power_{lead}'] = power_high
        newfeaturedata.loc[i,f'Power_Ratio_{lead}'] = power_ratio
        newfeaturedata.loc[i,f'Total_Harmonic_Power_{lead}'] = total_harmonic_power
        newfeaturedata.loc[i,f'Average_Harmonic_Power_{lead}'] = average_harmonic_power
        newfeaturedata.loc[i,f'Max_Harmonic_Power_{lead}'] = max_harmonic_power
        newfeaturedata.loc[i, f'Harmonic_Power_Ratio_{lead}'] = harmonic_power_ratio
        newfeaturedata.loc[i, f'Spectral_Entropy_{lead}'] = spectral_entropy
        print("completed:",round((i/12209)*100,3),"%")  # 顯示處理進度。
    
    # 保存每個導聯計算的特徵至CSV文件。
    newfeaturedata.to_csv(f"spectrum_training_lead{lead}.csv", index=False)
#%% 將每個取用導程的spectrum特徵合併
import numpy as np
import pandas as pd
featuredata0 = pd.read_csv("spectrum_featuredata_lead0.csv")

featuredata1 = pd.read_csv("spectrum_featuredata_lead1.csv")

featuredata4 = pd.read_csv("spectrum_featuredata_lead4.csv")

featuredata5 = pd.read_csv("spectrum_featuredata_lead5.csv")

featuredata9 = pd.read_csv("spectrum_featuredata_lead9.csv")

featuredata10 = pd.read_csv("spectrum_featuredata_lead10.csv")

featuredata11 = pd.read_csv("spectrum_featuredata_lead11.csv")

new_data = pd.concat([featuredata0, featuredata1, featuredata4, featuredata5, featuredata9, featuredata10, featuredata11], axis=1)

data = pd.DataFrame()
data = pd.concat([data, new_data], axis=1)
data.to_csv("spectrum_training_features.csv", index=False)
