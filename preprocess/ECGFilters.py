import scipy.io as sio
import numpy as np
import os
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import freqz
from scipy.signal import medfilt, iirnotch, butter, lfilter
import matplotlib.pyplot as plt


"""
Author: Wilson.Zhang
Date: 2019/03/21
Usage: Functions for ECG filters.
Reference: 
The ULg Multimodality Drowsiness Database (called DROZY) and Examples of Use.
"""


# Set Baseline Correction Filter...
# Reference: https://stackoverflow.com/questions/29156532/python-baseline-correction-library
def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


# Set diff function...
def differential(y):
    d = []
    for index in range(len(y)-1):
        tmp = y[index+1] - y[index]
        d.append(tmp)
    return np.array(d)


# Set IIR Notch filter...
def not_filter(ecg, freq=512.0, order=10.0):
    # power line frequency: 50Hz
    b, a = iirnotch(50.0, order, freq)
    y = lfilter(b, a, ecg)
    return np.asarray(y)


# Set FIR LPF filter...
def lpf_filter(ecg, freq=512.0, order=20.0):
    # according to review on ECG filter, fc: 150Hz or 40Hz.
    # We choose 32Hz.
    nyq = 0.5 * freq
    low = 32 / nyq
    b, a = butter(order, low, btype='lowpass', analog=False)
    y = lfilter(b, a, ecg)
    return np.asarray(y)


# Set Median filter...
def baseline_median_filter(ecg, freq=512):

    k_size_200ms = int(0.2 * freq)
    k_size_600ms = int(0.6 * freq)
    # kernel size must be odd
    if k_size_200ms % 2 != 1:
        k_size_200ms += 1
    if k_size_600ms % 2 != 1:
        k_size_600ms += 1
    result_200ms_med = medfilt(ecg, kernel_size=k_size_200ms)
    baseline = medfilt(result_200ms_med, kernel_size=k_size_600ms)

    return ecg - baseline, baseline


# Set FIR HPF filter...
def hpf_filter(ecg, freq=512.0, order=3.0):
    # according to review on ECG filter, fc: 0.67Hz
    nyq = 0.5 * freq
    high = 0.37 / nyq
    b, a = butter(order, high, btype='highpass', analog=False)
    y = lfilter(b, a, ecg)
    return np.asarray(y)


# Subplot func for filter comparison...
def subplotfunc(y1, y2=None, y3=None, y4=None, num=1, freq=512, freq2=512, freq3=512, freq4=512, per=1.0):
    plt.figure(1)
    plt.subplot(num, 1, 1)
    y1 = y1[:int(len(y1)*per)]
    plt.plot(np.arange(len(y1)) / freq, y1)
    plt.title("ECG preprocessing")
    plt.xlim([0, len(y1) / freq])
    plt.xlabel("Time (s)")

    if y2 is not None:
        plt.subplot(num, 1, 2)
        y2 = y2[:int(len(y2)*per)]
        plt.plot(np.arange(len(y2)) / freq2, y2)
        # plt.title("Y2")
        plt.xlim([0, len(y2) / freq2])
        plt.xlabel("Time (s)")

    if y3 is not None:
        plt.subplot(num, 1, 3)
        y3 = y3[:int(len(y3)*per)]
        plt.plot(np.arange(len(y3)) / freq3, y3)
        # plt.title("Y3")
        plt.xlim([0, len(y3) / freq3])
        plt.xlabel("Time (s)")

    if y4 is not None:
        plt.subplot(num, 1, 4)
        y4 = y4[:int(len(y4)*per)]
        plt.plot(np.arange(len(y4)) / freq4, y4)
        # plt.title("Y4")
        plt.xlim([0, len(y4) / freq4])
        plt.xlabel("Time (s)")

    plt.show()


# FFT for drawing frequency
def spectrum(ecg, fs=512, fft_start=1000, fft_size=4000):
    xs = ecg[fft_start: fft_start + fft_size]
    xf = np.fft.rfft(xs) / fft_size
    freq = np.linspace(0, fs/2, fft_size//2 + 1)
    xfp = 100 + 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    # print("shape of xfp:", xfp.shape, "length of freq:", freq.shape)
    return xfp[:xfp.shape[0] // 2], freq[:freq.shape[0] // 2]


def main():
    # switch working folder to main path
    wf = os.getcwd()
    # print(wf)
    if "preprocess" in wf:
        os.chdir("../")

    if not os.path.exists('./dataset'):
        exit()
    if not os.path.exists('./dataset/Filtered'):
        os.mkdir('./dataset/Filtered')

    for i in range(1, 15):
        for j in range(1, 4):
            ecgs_filtered = []
            val = "ECG{}_{}".format(i, j)
            try:
                data = sio.loadmat("./dataset/Original/{}.mat".format(val))[val]
            except IOError:
                print("{} doesn't exist.".format(val))
                continue
            else:
                print("Start filtering {}".format(val))
                for k in range(len(data)):
                    y = data[k]
                    tmp = not_filter(y)
                    tmp2 = lpf_filter(tmp)
                    ecg, _ = baseline_median_filter(tmp2)
                    ecgs_filtered.append(ecg)
                np.save("./dataset/Filtered/Filtered-{}-{}".format(i, j), ecgs_filtered)
                print("{} finished & saved".format(val))


# Try filters
if __name__ == "__main__":
    main()
