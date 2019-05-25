import numpy as np
import os
import matplotlib.pyplot as plt
from multiprocessing import Process

"""
Author: Wilson Zhang
Date: 2019/04/18
Usage: Employ Short Time Fourier Transform to ECG Signals and save
spectrum map as fig.
Attention: multithread func is employed to speed up!
"""


# switch working folder to main path
wf = os.getcwd()
# print(wf)
if "preprocess" in wf:
    os.chdir("../")
# print(os.getcwd())

if not os.path.exists("./dataset/Specgrams"):
    os.makedirs("./dataset/Specgrams")
if not os.path.exists("./dataset/Specgrams/1"):
    os.makedirs("./dataset/Specgrams/1")
if not os.path.exists("./dataset/Specgrams/0"):
    os.makedirs("./dataset/Specgrams/0")
if not os.path.exists("./dataset/Specgrams/-1"):
    os.makedirs("./dataset/Specgrams/-1")

dirs = "./dataset/Resampled/"

nfft = 200
fs = 200


def main_true():
    # load true ecg data
    num = 1
    for i in range(1, 15):
        val = "{}-1".format(i)
        try:
            data = np.load("{}resampled-{}.npy".format(dirs, val))
        except IOError:
            print("{} doesn't exist.".format(val))
            continue
        else:
            print("Start processing {}".format(val))
            for k in range(len(data)):
                y = data[k]
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                # spectrogram param keeps the same to Matlab. (include the window if using hamming)
                plt.axis("off")
                plt.specgram(y, NFFT=nfft, Fs=fs, noverlap=int(fs*(475/512)), window=np.bartlett(fs))
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                # save fig without the white border.
                plt.savefig("./dataset/Specgrams/1/{}.png".format(num), bbox_inches=extent)
                plt.close()
                num += 1
            print("{} finished. pic num is {}".format(val, num-1))


def main_false():
    # load false ecg data
    num = 1
    for t in range(1, 15):
        val = "{}-3".format(t)
        try:
            data = np.load("{}resampled-{}.npy".format(dirs, val))
        except IOError:
            print("{} doesn't exist.".format(val))
            continue
        else:
            print("Start processing {}".format(val))
            for k in range(len(data)):
                y = data[k]
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                # spectrogram param keeps the same to Matlab. (include the window if using hamming)
                plt.axis("off")
                plt.specgram(y, NFFT=nfft, Fs=fs, noverlap=int(fs * (475 / 512)), window=np.bartlett(fs))
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                # save fig without the white border.
                plt.savefig("./dataset/Specgrams/0/{}.png".format(num), bbox_inches=extent)
                plt.close()
                num += 1
            print("{} finished. pic num is {}".format(val, num-1))


def main_unlabel():
    # load unlabel ecg data
    num = 1
    for x in range(1, 15):
        val = "{}-2".format(x)
        try:
            data = np.load("{}resampled-{}.npy".format(dirs, val))
        except IOError:
            print("{} doesn't exist.".format(val))
            continue
        else:
            print("Start processing {}".format(val))
            for k in range(len(data)):
                y = data[k]
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                # spectrogram param keeps the same to Matlab. (include the window if using hamming)
                plt.axis("off")
                plt.specgram(y, NFFT=nfft, Fs=fs, noverlap=int(fs * (475 / 512)), window=np.bartlett(fs))
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                # save fig without the white border.
                plt.savefig("./dataset/Specgrams/-1/{}.png".format(num), bbox_inches=extent)
                plt.close()
                num += 1
            print("{} finished. pic num is {}".format(val, num-1))


if __name__ == "__main__":
    p1 = Process(target=main_true)
    p1.start()
    p2 = Process(target=main_false)
    p2.start()
    p3 = Process(target=main_unlabel)
    p3.start()
