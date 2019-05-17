import numpy as np
import scipy.io as sio
# from ECGFilters import subplotfunc
from scipy.signal import resample
import os

"""
Author: Wilson Zhang
Date: 2019/04/11
Usage: Downsample ECG Signals to 200Hz.
Signals will be saved to folder 'Resampled' !!!!
"""


def main():
    # switch working folder to main path
    wf = os.getcwd()
    # print(wf)
    if "preprocess" in wf:
        os.chdir("../")
    # print(os.getcwd())

    dirs = "./dataset"

    if not os.path.exists("./dataset/Resampled"):
        os.makedirs("./dataset/Resampled")

    for i in range(14):
        for j in range(3):
            ecgs_resampled = []
            val = "{}-{}".format(i+1, j+1)
            try:
                data = np.load("{}/Analysed/analysed-{}.npy".format(dirs, val))
            except IOError:
                print(val + " doesn't exist.")
                continue
            else:
                print("Start resampling {}".format(val))
                for k in range(len(data)):
                    y = data[k]
                    res = resample(y, 6000)
                    # subplotfunc(y1=y, y2=res, num=2, freq=512, freq2=200)
                    ecgs_resampled.append(res)
                print(str(len(ecgs_resampled)))
                np.save("{}/Resampled/resampled-{}".format(dirs, val), ecgs_resampled)
                print("{} finished & saved.".format(val))


def remove_bad_signals():
    # switch working folder to main path
    wf = os.getcwd()
    # print(wf)
    if "preprocess" in wf:
        os.chdir("../")
    # print(os.getcwd())

    dirs = "./dataset"

    if not os.path.exists("./dataset/Analysed-original"):
        os.makedirs("./dataset/Analysed-original")

    # load missnum files...
    ecg0 = []
    ecg1 = []
    for i in range(1, 15):
        for j in range(1, 4):
            val = "{}-{}".format(i, j)
            try:
                data = sio.loadmat("{}/Original/ECG{}_{}.mat".format(dirs, i, j))["ECG{}_{}".format(i, j)]
            except IOError:
                print("{} doesn't exist, skipped.".format(val))
                continue
            else:
                try:
                    y = np.load("{}/Missnum/missnum-{}.npy".format(dirs, val))
                except IOError:
                    print("All signal accepted in {}.".format(val))
                    np.save("{}/Analysed-original/original-{}.npy".format(dirs, val), data)
                    if j == 1 and len(ecg1) == 0:
                        ecg1 = data
                    elif j == 1 and len(ecg1) != 0:
                        ecg1 = np.append(ecg1, data, axis=0)
                    elif j == 3 and len(ecg0) == 0:
                        ecg0 = data
                    elif j == 3 and len(ecg0) != 0:
                        ecg0 = np.append(ecg0, data, axis=0)
                    print("{} finished & converted.".format(val))
                    continue
                else:
                    if (39 - len(y)) == len(data):
                        print("in {}, unaccepted signals have been removed, skipped.".format(val))
                        np.save("{}/Analysed-original/original-{}.npy".format(dirs, val), data)
                        if j == 1 and len(ecg1) == 0:
                            ecg1 = data
                        elif j == 1 and len(ecg1) != 0:
                            ecg1 = np.append(ecg1, data, axis=0)
                        elif j == 3 and len(ecg0) == 0:
                            ecg0 = data
                        elif j == 3 and len(ecg0) != 0:
                            ecg0 = np.append(ecg0, data, axis=0)
                        print("{} finished & converted.".format(val))
                        continue
                    print("num {} will be removed".format(y))
                    data = np.delete(data, y, axis=0)
                    print("new size: {}".format(data.shape))
                    if data.shape[0] != 0:
                        np.save("{}/Analysed-original/original-{}.npy".format(dirs, val), data)
                        if j == 1 and len(ecg1) == 0:
                            ecg1 = data
                        elif j == 1 and len(ecg1) != 0:
                            ecg1 = np.append(ecg1, data, axis=0)
                        elif j == 3 and len(ecg0) == 0:
                            ecg0 = data
                        elif j == 3 and len(ecg0) != 0:
                            ecg0 = np.append(ecg0, data, axis=0)
                    print("{} finished & converted.".format(val))
    # saving mat for specgram function in matlab...
    sio.savemat("{}/Analysed-original/ECG1.mat".format(dirs), {"ECG1": ecg1})
    sio.savemat("{}/Analysed-original/ECG0.mat".format(dirs), {"ECG0": ecg0})


if __name__ == "__main__":
    remove_bad_signals()
