import numpy as np
import scipy.io as sio
import math
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import os


"""
Author: Wilson Zhang & Pili Li
Date: 2018/12/13
Usage: Detecting QRS complexities based on Pan-Tompkins Algorithm
Reference: https://github.com/KChen89/QRS-detection
"""


def load_ecg(name, via, col=None):
    # caution: start with 0!
    if col is not None:
        return [sio.loadmat(name)[via][col]]
    else:
        return sio.loadmat(name)[via]


def butter_bandpass(lowcut, highcut, freq, order=5):
    nyq = 0.5 * freq
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def fir_bandpass_filter(ecg, lowcut, highcut, freq, order=5):
    """
    Usage: bandpass filter for ECG signals
    :param ecg: ECG signal in numpy format.
    :param lowcut: frequency for highpass FIR filter.
    :param highcut: frequency for lowpass FIR filter.
    :param freq: signal frequency.
    :param order: filter order, higher will be better.
    :return: signal has been filtered. (in ndarray form)
    """
    b, a = butter_bandpass(lowcut, highcut, freq, order=order)
    y = lfilter(b, a, ecg)
    return np.asarray(y)


# 对信号作微分后求平方。
# Plan A
def diffsqr_trans(ecg, ws):
    """
    Usage: see comments above.
    :param ecg: ECG signal has been filtered. (in ndarray form).
    :param ws: width for differential function.
    :return: signal after diff & square. (in ndarray form).
    """
    lgth = len(ecg)
    diff = np.zeros(lgth)
    ecg = np.pad(ecg, ws, 'edge')
    for i in range(lgth):
        temp = ecg[i:i+ws+ws+1]  # 取消了填充造成的延迟
        left = temp[ws] - temp[0]  # 数组左起
        right = temp[ws] - temp[-1]  # 数组右起
        diff[i] = min(left, right)
        diff[diff < 0] = 0  # 数组中小于零的值取零
        # 将x-ws与x+ws的斜率进行对比，随后去除波形中小于目标斜率的波形。
    return np.multiply(diff, diff)


# 积分
def integrate(ecg, ws):
    lgth = len(ecg)
    integrate_ecg = np.zeros(lgth)
    ecg = np.pad(ecg, math.ceil(ws/2), mode='symmetric')
    for i in range(lgth):
        integrate_ecg[i] = np.sum(ecg[i:i+ws])/ws
    return integrate_ecg


# 定位开始
# 中间通过求对应区间的方差以达到确定
def find_peak(data, ws, threshhold=0.005):
    # 开始定位前需要对信号归一化以便于调参。
    data = (data - min(data)) / (max(data) - min(data))
    lgth = len(data)
    true_peaks = []
    for i in range(lgth-ws+1):
        temp = data[i:i+ws]
        if np.var(temp) < threshhold:
            continue
        # else:
        #     print(np.var(temp))
        index = int((ws-1)/2)
        peak = True
        for j in range(index):
            if temp[index-j] <= temp[index-j-1] or temp[index+j] <= temp[index+j+1]:
                peak = False
                break

        if peak is True:
            true_peaks.append(i+index)
    return np.asarray(true_peaks)


def find_r_peaks(ecg, peaks, ws, freq=512):
    num_peak = len(peaks)
    r_peaks = []
    ture_r_peaks = []
    for index in range(num_peak):
        i = peaks[index]
        if i-2*ws > 0 and i < len(ecg):
            temp_ecg = ecg[i-2*ws:i]
            r_peaks.append(np.argmax(temp_ecg)+i-2*ws)
    for index in range(len(r_peaks)-1):
        r_pre = r_peaks[index]
        r_now = r_peaks[index + 1]
        tmp = (r_now-r_pre)/512
        if tmp >= 0.4:
            ture_r_peaks.append(r_now)
        else:
            continue

    return np.asarray(ture_r_peaks)


def plot_r_peaks(ecg, r_peaks, freq=512):
    index = np.arange(len(ecg))/freq
    plt.figure()
    plt.plot(index, ecg, "b", label="ECG Signal")
    plt.plot(r_peaks/freq, ecg[r_peaks], "ro", label="R peaks")
    plt.xlim([0, len(ecg)/freq])
    # plt.plot(range(len(ecg)), ecg, "b", label="ECG Signal")
    # plt.plot(r_peaks, ecg[r_peaks], "ro", label="R peaks")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()


def adjust_threshold(i, ecg, ecg_filtered, r_peaks, ecg_integrate, tmp, freq=512):
    """

    :param i: The number of ECG Signal (start with 0)
    :param ecg: ECG signal
    :param ecg_filtered: ECG signal been filtered
    :param r_peaks: R peaks extracted by PT Algorithm
    :param ecg_integrate: ECG Signal which has been integrated
    :param tmp: threshold
    :param freq: sample freq
    :return: 0: fail to extract R peaks.
    """
    try:
        plot_r_peaks(ecg, r_peaks)
    except IndexError:
        print("No." + str(i + 1) + " can't extract R peaks, mark as noise.")
        return 0, tmp
    else:
        _word = "Is No." + str(i + 1) + " R-R interval accurate? " \
                                       "\n[1]: Yes " \
                                       "\n[2]: Need Lower Threshhold (miss R waves) " \
                                       "\n[3]: Need Higher Threshhold (miscount waves)" \
                                       "\n[0]: Noise. \n"
        _res = input(_word)
        try:
            _res = int(_res)
        except ValueError:
            _res = 4

        while _res != 1:
            if _res == 0:
                # return noise sample
                return 0, tmp

            elif _res == 2:
                tmp = tmp / 1.5
                _peaks = find_peak(ecg_integrate, int(freq/5), tmp)
                _r_peaks = find_r_peaks(ecg_filtered, _peaks, int(freq/40))
                plot_r_peaks(ecg, _r_peaks)
                _res = input(_word)
                try:
                    _res = int(_res)
                except ValueError:
                    _res = 4

            elif _res == 3:
                tmp = tmp * 1.5
                _peaks = find_peak(ecg_integrate, int(freq / 5), tmp)
                _r_peaks = find_r_peaks(ecg_filtered, _peaks, int(freq / 40))
                plot_r_peaks(ecg, _r_peaks)
                _res = input(_word)
                try:
                    _res = int(_res)
                except ValueError:
                    _res = 4

            else:
                _res = input(_word)
                try:
                    _res = int(_res)
                except ValueError:
                    _res = 4

        else:
            return 1, tmp


def rr_detection(name, freq=512, var=None, col=None, switchauto=False):
    """
    feature unfinished:
    1. load .mat file & read target variable
    2. locate R-R interval and display them
    3. fine tune & relocate & deposit
    4. save R-R interval.
    :param name: ECG signals (more than 1 sample is allowed, should be in ndarray.).
    :param var: target variable (not used now)
    :param freq: sample frequency for signal
    :param col: column of ECG (not used now)
    :param switchauto: whether manually judge R-R interval extraction quality.
    :return: status: 0: i/o failure
                     1: done
                     -1: no R-R interval detected.
             ns: noise sample sequence (caution! start with 0)
             rr_intervals: R-R intervals
             ecgs: ecg signals unfiltered
             ecgs_filtered: ecg signals filtered
    """
    resdict = {"status": 0, "ns": [], "rr_intervals": [], "ecgs": [], "ecgs_filtered": []}
    data = name
    ns = []
    rr_intervals = []
    ecgs = []
    ecgs_filtered = []
    tmp = 0.005
    auto = False
    for i in range(len(data)):
        ecg = data[i]
        ecg_filtered = fir_bandpass_filter(ecg, 5, 12, freq, 5)

        ecg_trans = diffsqr_trans(ecg_filtered, int(freq/20))

        # plt.figure(2)
        ws = int(freq/8)
        ecg_integrate = integrate(ecg_trans, ws)/ws
        ws = int(freq/6)
        ecg_integrate = integrate(ecg_integrate, ws)/ws
        ws = int(freq/36)
        ecg_integrate = integrate(ecg_integrate, ws)/ws
        ws = int(freq/72)
        ecg_integrate = integrate(ecg_integrate, ws)/ws

        peaks = find_peak(ecg_integrate, int(freq/5), tmp)
        r_peaks = find_r_peaks(ecg_filtered, peaks, int(freq/40))
        if auto is False:
            state, tmp = adjust_threshold(i, ecg, ecg_filtered, r_peaks, ecg_integrate, tmp)
            if state == 0:
                ns.append(i)
                # close auto process
                auto = False
                continue
            else:
                # fetch r_peaks with new threshold
                peaks = find_peak(ecg_integrate, int(freq/5), tmp)
                r_peaks = find_r_peaks(ecg_filtered, peaks, int(freq/40))
                # change r peaks into rr interval.
                rr_interval = []
                for index in range(len(r_peaks) - 1):
                    rr_interval.append((r_peaks[index + 1] - r_peaks[index]))

                # wdnmd, 注意缩进!!!!!!!
                # save signal & rr interval into list.
                rr_intervals.append(rr_interval)
                ecgs.append(ecg)
                ecgs_filtered.append(ecg_filtered)

                # turn auto process on if allowed...
                if switchauto is True:
                    auto = True
                    print("Auto Process is on...")
                continue
        else:

            # auto process is on...
            # directly change r peaks into rr interval...
            rr_interval = []
            print("No.{}".format(i+1))
            if len(r_peaks) <= 10:
                print("Oops, No.{} was failed to extract r peaks. Auto process is switched off and marked as noise.".format(i + 1))
                ns.append(i)
                auto = False
                continue
            for index in range(len(r_peaks) - 1):
                # Check R-R interval value...
                if (r_peaks[index + 1] - r_peaks[index]) <= freq * 0.25 or (
                        r_peaks[index + 1] - r_peaks[index]) >= freq * 2:
                    print("Oops, No.{} was failed to extract r peaks. Auto process is switched off.".format(i+1))
                    rr_interval = []
                    auto = False
                    # then redo adjust threshold process
                    state, tmp = adjust_threshold(i, ecg, ecg_filtered, r_peaks, ecg_integrate, tmp)
                    if state == 0:
                        # close auto process
                        auto = False
                        break  # leave this loop
                    else:
                        # fetch r_peaks with new threshold
                        peaks = find_peak(ecg_integrate, int(freq / 5), tmp)
                        r_peaks = find_r_peaks(ecg_filtered, peaks, int(freq / 40))
                        # change r peaks into rr interval.
                        rr_interval = []
                        for index_2 in range(len(r_peaks) - 1):
                            rr_interval.append((r_peaks[index_2 + 1] - r_peaks[index_2]))

                        # turn auto process on if allowed...
                        print("Alright, problem solved.")
                        if switchauto is True:
                            auto = True
                        break  # leave this loop!
                else:
                    rr_interval.append(r_peaks[index + 1] - r_peaks[index])

            # wdnmd, 注意缩进!!!!!!!
            # save signal & rr interval into list if signal is not noise.
            if auto is True:
                rr_intervals.append(rr_interval)
                ecgs.append(ecg)
                ecgs_filtered.append(ecg_filtered)
                continue
            else:
                ns.append(i)
                # close auto process!
                continue

    # print("{} is finished.".format(name))
    if len(ecgs) is not 0:
        resdict["status"] = 1
        resdict["ns"] = ns
        resdict["rr_intervals"] = rr_intervals
        resdict["ecgs"] = ecgs
        resdict["ecgs_filtered"] = ecgs_filtered
    return resdict


def main():
    # switch working folder to main path
    wf = os.getcwd()
    # print(wf)
    if "preprocess" in wf:
        os.chdir("../")
    if not os.path.exists("./dataset/Analysed"):
        os.makedirs("./dataset/Analysed")

    if not os.path.exists("./dataset/Missnum"):
        os.makedirs("./dataset/Missnum")

    if not os.path.exists("./dataset/RRIs"):
        os.makedirs("./dataset/RRIs")

    dirs = "./dataset"
    name = str(input("please entry the patient number:\n"))
    num = str(input("please entry the series number:\n"))
    val = "{}-{}".format(name, num)
    try:
        os.remove('{}/Missnum/missnum-{}.npy'.format(dirs, val))
        os.remove('{}/Analysed/analysed-{}.npy'.format(dirs, val))
        os.remove('{}/RRIs/rr-{}.npy'.format(dirs, val))
    except IOError:
        print("{} cleaned".format(val))
    else:
        print("{} cleaned".format(val))
    try:
        data = np.load("{}/Filtered/filtered-{}.npy".format(dirs, val))
    except IOError:
        print("File doesn't exist")
    else:
        print("Start retrieving {}".format(val))
        # start quality analysis...
        resdict = rr_detection(data, switchauto=True)
        if resdict["status"] is 1 and len(resdict["rr_intervals"]) is not 0:
            np.save("{}/RRIs/rr-{}".format(dirs, val), resdict['rr_intervals'])
            np.save("{}/Analysed/analysed-{}".format(dirs, val), resdict['ecgs'])
            if len(resdict["ns"]) is not 0:
                np.save("{}/Missnum/missnum-{}".format(dirs, val), resdict['ns'])
        else:
            np.save("{}/Missnum/missnum-{}".format(dirs, val), range(39))  # missing num start with 0!
        print("{} finished & saved.".format(val))


if __name__ == "__main__":
    main()
