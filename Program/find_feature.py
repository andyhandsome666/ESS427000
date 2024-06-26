#%%
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

#%%
def denoise(data, order=4, lowcut=20, fs=500):
    """
    The low pass Butterworth filter written by TA.
    """
    b, a = sig.butter(order, lowcut, 'lowpass', fs=fs)
    sig_denoised = sig.filtfilt(b, a, data)
    return sig_denoised

def find_and_filter_r_peaks(data, distance=150, height=None, threshold=0.5):
    """
    Find and filter R peaks in the ECG signal.

    Parameters:
    - data: Input ECG signal data.
    - distance: Minimum distance between peaks.
    - height: Required height of peaks.
    - threshold: Relative threshold for filtering peaks based on maximum peak.

    Returns:
    - filtered_peaks: Array of filtered R peaks.
    """
    peaks, _ = sig.find_peaks(data, distance=distance, height=height)
    if len(peaks) == 0:
        return peaks

    pmax = max(data[peaks])
    peaks = [peak for peak in peaks if data[peak] >= threshold * pmax]
    filtered_peaks = []
    for i in range(len(peaks) - 1):
        if peaks[i + 1] - peaks[i] >= distance:
            filtered_peaks.append(peaks[i])
    filtered_peaks.append(peaks[-1])

    return np.array(filtered_peaks)

def find_pqrst_peaks(data, r_peaks):
    """
    Identify P, Q, R, S, T, S', T', P', L' points in the ECG signal.

    Parameters:
    - data: Input ECG signal data.
    - r_peaks: Array of R peaks.

    Returns:
    - features: Dictionary containing identified P, Q, R, S, T, S', T', P', L' points.
    """
    features = {
        'P': [], 'Q': [], 'R': r_peaks, 'S': [], 'T': [],
        "S'": [], "T'": [], "P'": [], "L'": []
    }

    for i in range(len(r_peaks) - 1):
        # Find T peak
        t_range = np.arange(r_peaks[i] + 30, (r_peaks[i] + r_peaks[i + 1]) // 2)
        t_peak = t_range[0] + np.argmax(data[t_range])
        features['T'].append(t_peak)
        
        # Find P peak
        p_range = np.arange((r_peaks[i] + r_peaks[i + 1]) // 2 + 30, r_peaks[i + 1] - 30)
        p_peak = p_range[0] + np.argmax(data[p_range])
        features['P'].append(p_peak)
        
        # Find S peak
        s_range = np.arange(r_peaks[i], t_peak)
        s_peak = s_range[0] + np.argmin(data[s_range])
        features['S'].append(s_peak)
        
        # Find Q peak
        q_range = np.arange(p_peak, r_peaks[i + 1])
        q_peak = q_range[0] + np.argmin(data[q_range])
        features['Q'].append(q_peak)
        
        # Find inflection points and slope change points
        for name, point_range, comparison in [
            ("S'", np.arange(t_peak, s_peak, -1), lambda j: data[j] * data[j - 1] < 0),
            ("T'", np.arange(t_peak, t_peak + 200), lambda j: data[j] * data[j + 1] < 0),
            ("L'", np.arange(p_peak, p_peak - 200, -1), lambda j: data[j] * data[j - 1] < 0),
            ("P'", np.arange(p_peak, q_peak), lambda j: data[j] * data[j + 1] < 0)
        ]:
            found = False
            min_slope_index = -1
            min_slope = float('inf')
            for j in point_range:
                if comparison(j):
                    features[name].append(j)
                    found = True
                    break
                # Calculate the slope
                slope = abs(data[j] - data[j - 1])
                if slope < min_slope:
                    min_slope = slope
                    min_slope_index = j
            if not found and min_slope_index != -1:
                features[name].append(min_slope_index)
            if not found and min_slope_index == -1:
                default_point = (t_peak + s_peak) // 2 if name == "S'" else \
                                t_peak + 100 if name == "T'" else \
                                p_peak - 200 if name == "L'" else \
                                (p_peak + q_peak) // 2
                features[name].append(default_point)
    return features

#If you want to visualize the {PQRSTP'L'S'T'} points on ecg then uncomment the following code!
"""
#%%
# Load the ECG data from a .npy file
data = np.load('ML_Train.npy', mmap_mode='r')
#%%
# Select the first ECG signal from the data
ecg_data = data[0, 0, :]
# Apply denoising to the ECG signal
sig_denoised = denoise(ecg_data)
# Find and filter R peaks in the denoised ECG signal
r_peaks = find_and_filter_r_peaks(sig_denoised, distance=150)
# Find P, Q, R, S, T, S', T', P', L' points in the ECG signal
features = find_pqrst_peaks(sig_denoised, r_peaks)
# Plot the denoised ECG signal with identified PQRST points
plt.figure(figsize=(15, 7))
plt.plot(sig_denoised, linewidth=0.5, label="Denoised ECG")
plt.scatter(features['P'], sig_denoised[features['P']], color='green', label='P peaks')
plt.scatter(features['Q'], sig_denoised[features['Q']], color='red', label='Q peaks')
plt.scatter(features['R'], sig_denoised[features['R']], color='blue', label='R peaks')
plt.scatter(features['S'], sig_denoised[features['S']], color='purple', label='S peaks')
plt.scatter(features['T'], sig_denoised[features['T']], color='orange', label='T peaks')
plt.scatter(features["S'"], sig_denoised[features["S'"]], color='pink', label="S' points", marker='o')
plt.scatter(features["T'"], sig_denoised[features["T'"]], color='cyan', label="T' points", marker='o')
plt.scatter(features["P'"], sig_denoised[features["P'"]], color='brown', label="P' points", marker='o')
plt.scatter(features["L'"], sig_denoised[features["L'"]], color='yellow', label="L' points", marker='o')
plt.legend()
plt.title('Denoised ECG with PQRST Points and Slope Change Points')
plt.show()
# %%
"""