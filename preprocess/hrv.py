import numpy as np
from hrvanalysis.extract_features import get_time_domain_features, get_frequency_domain_features
import os
from QRSdetection import rr_detection


"""
Author: Wilson Zhang
Date: 2019/05/20
Usage: Extract HRV based time and frequency domain features from short-time ECG Signals
Attention: errors between Matlab and Python left unsolved.
"""


def hrv_feature_extraction(inputs, fs=512):
    """

    :param inputs: R-R intervals as numpy array
    :param fs: ECG sample freq for regularization
    :return: list contains 9 features, listed as follow.
    """
    # features = {"HR": float,
    #             "SDNN": float,
    #             "RMSSD": float,
    #             "SDSD": float,
    #             "NN50": float,
    #             "PNN50": float,
    #             "LF": float,
    #             "HF": float,
    #             "LFHF": float}
    features = []
    if inputs is []:
        return features
    else:
        # regularization to ms
        inputs = np.true_divide(inputs, fs)
        inputs = np.multiply(inputs, 1000)
        tmp = get_time_domain_features(inputs)
        tmp2 = get_frequency_domain_features(inputs, method='lomb')
        features.append(tmp['mean_hr'])
        features.append(tmp['sdnn'])
        features.append(tmp['rmssd'])
        features.append(tmp['sdsd'])
        features.append(tmp['nni_50'])
        features.append(tmp['pnni_50'])  # in percentage
        features.append(tmp2['lf'])
        features.append(tmp2['hf'])
        features.append(tmp2['lf_hf_ratio'])
        return features


def main():
    # switch working folder to main path
    wf = os.getcwd()
    # print(wf)
    if "preprocess" in wf:
        os.chdir("../")

    dirs = './dataset'
    freq = 512
    if not os.path.exists('./dataset/RRIs'):
        exit()  # no rr intervals, exit.

    if not os.path.exists('./dataset/HRV'):
        os.makedirs('./dataset/HRV')
    if not os.path.exists('./dataset/HRV/1'):
        os.makedirs('./dataset/hrv/1')
    if not os.path.exists('./dataset/HRV/0'):
        os.makedirs('./dataset/hrv/0')
    if not os.path.exists('./dataset/HRV/-1'):
        os.makedirs('./dataset/HRV/-1')

    num = 1
    for i in range(1, 15):
        val = '{}-1'.format(i)
        try:
            data = np.load('{}/RRIs/rr-{}.npy'.format(dirs, val))
            ecgs = np.load('{}/Analysed/analysed-{}.npy'.format(dirs, val))
        except IOError:
            print("{} doesn't exist.".format(val))
            continue
        else:
            print("Start processing {}".format(val))
            if len(data) != len(ecgs):
                print("不用跑了，都没一一对应，重跑评估脚本")
            for k in range(len(data)):
                y = data[k]
                # try to repair r-r interval first.
                if len(y) <= 15:  # r-r interval should be more than 15.
                    tmp = rr_detection(name=[ecgs[k]])
                    if tmp['status'] is not 1:
                        print("你在玩锤子呢？")
                        exit()
                    else:
                        y = tmp['rr_intervals'][0]
                        print(y)
                features = hrv_feature_extraction(inputs=y, fs=freq)
                if len(features) is not 0:
                    np.save('{}/HRV/1/{}'.format(dirs, num), features)
                    num += 1
            print("{} finished, the num of samples is {}.".format(val, num-1))

    num = 1
    for i in range(1, 15):
        val = '{}-3'.format(i)
        try:
            data = np.load('{}/RRIs/rr-{}.npy'.format(dirs, val))
            ecgs = np.load('{}/Analysed/analysed-{}.npy'.format(dirs, val))
        except IOError:
            print("{} doesn't exist.".format(val))
            continue
        else:
            print("Start processing {}".format(val))
            if len(data) != len(ecgs):
                print("不用跑了，都没一一对应，重跑评估脚本")
            for k in range(len(data)):
                y = data[k]
                # try to repair r-r interval first.
                if len(y) <= 15:  # r-r interval should be more than 15.
                    tmp = rr_detection(name=[ecgs[k]])
                    if tmp['status'] is not 1:
                        print("你在玩锤子呢？")
                        exit()
                    else:
                        y = tmp['rr_intervals'][0]
                        print(y)
                features = hrv_feature_extraction(inputs=y, fs=freq)
                if len(features) is not 0:
                    np.save('{}/HRV/0/{}'.format(dirs, num), features)
                    num += 1
            print("{} finished, the num of samples is {}.".format(val, num-1))

    num = 1
    for i in range(1, 15):
        val = '{}-2'.format(i)
        try:
            data = np.load('{}/RRIs/rr-{}.npy'.format(dirs, val))
            ecgs = np.load('{}/Analysed/analysed-{}.npy'.format(dirs, val))
        except IOError:
            print("{} doesn't exist.".format(val))
            continue
        else:
            print("Start processing {}".format(val))
            if len(data) != len(ecgs):
                print("不用跑了，都没一一对应，重跑评估脚本")
            for k in range(len(data)):
                y = data[k]
                # try to repair r-r interval first.
                if len(y) <= 15:  # r-r interval should be more than 15.
                    tmp = rr_detection(name=[ecgs[k]])
                    if tmp['status'] is not 1:
                        print("你在玩锤子呢？")
                        exit()
                    else:
                        y = tmp['rr_intervals'][0]
                        print(y)
                features = hrv_feature_extraction(inputs=y, fs=freq)
                if len(features) is not 0:
                    np.save('{}/HRV/-1/{}'.format(dirs, num), features)
                    num += 1
            print("{} finished, the num of samples is {}.".format(val, num-1))


if __name__ == "__main__":
    main()
