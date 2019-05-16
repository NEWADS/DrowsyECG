import numpy as np
import os
from sklearn.manifold import Isomap


def shuffle(x, y):
    """
    function "shuffle"

    input:  feature vectors as 'x' and label as 'y'
    output: shuffled data as 'x_out' and correspond lable as 'y_out'

    """
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    x_out = x[index]
    y_out = y[index]
    return x_out, y_out


def data_generator(dirs):
    """

    :param dirs: data dir, which should have sub folders named "1", "0", "-1".
    :return: formatted dataset.
    """
    feature0 = []
    feature1 = []

    if not os.path.exists(dirs) and os.path.exists("{}/1".format(dirs)) and os.path.exists("{}/0".format(dirs)):
        print("no data detected.")
        return feature1, feature0

    # loading True dataset...
    contentlist = os.listdir("{}/1".format(dirs))
    contentlist.sort(key=lambda x: int(x[:len(x)-4]))  # 按升序排序
    for i in contentlist:
        # for stft & hrv
        if ".npy" in i:
            data = np.load("{}/1/{}".format(dirs, i))
            feature1.append(data)

    # loading False dataset...
    contentlist = os.listdir("{}/0".format(dirs))
    contentlist.sort(key=lambda x: int(x[:len(x)-4]))  # 按升序排序
    for i in contentlist:
        # for stft & hrv
        if ".npy" in i:
            data = np.load("{}/0/{}".format(dirs, i))
            feature0.append(data)

    if len(feature0) and len(feature1) is not 0:
        feature0 = np.asarray(feature0)
        feature1 = np.asarray(feature1)
        if len(feature0.shape) >= 3:
            feature0 = feature0.reshape(feature0.shape[0], feature0.shape[2])
        if len(feature1.shape) >= 3:
            feature1 = feature1.reshape(feature1.shape[0], feature1.shape[2])
        print("True dataset size: {}. \nFalse dataset size: {}.".format(feature1.shape, feature0.shape))
        return feature1, feature0
    else:
        print("failed to load dataset & dirs empty, please check your data")
        return feature1, feature0


def isomap(feature1, feature0):
    """
    :param feature1: true transfer learning dataset
    :param feature0: false transfer learning dataset
    :return: feature1 and feature0 dataset numpy ndarray.
    """
    drr_8 = Isomap(n_components=8)
    feature1 = drr_8.fit_transform(feature1)
    drr_8 = Isomap(n_components=8)
    feature0 = drr_8.fit_transform(feature0)
    print("Isomap is finished. \nTrue dataset size: {}. \nFalse dataset size: {}.".format(feature1.shape, feature0.shape))
    return np.asarray(feature1), np.asarray(feature0)


if __name__ == "__main__":
    f1, f0 = data_generator("./dataset/Transferred/icp")
    isomap(f1, f0)
