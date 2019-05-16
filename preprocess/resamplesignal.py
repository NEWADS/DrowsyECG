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


if __name__ == "__main__":
    main()
