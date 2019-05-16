import tensorflow as tf
import numpy as np
import os
import argparse
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.applications import InceptionV3
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.inception_v3 import preprocess_input

parser = argparse.ArgumentParser(description="transfer learning feature extraction model for this project")
parser.add_argument("--model", required=True, type=int, choices=[0, 1], metavar="MODEL",
                    help="model used for feature extraction, 1 is Inception_V3, 0 is VGG_16")
args = parser.parse_args()
X = args.model


def transfer_feature_extraction(img, model="Inception_V3"):
    """

    :param img: stft image
    :param model: which model to use
    :return: model prediction (in numpy ndarray)
    """
    data = image.img_to_array(img)
    data = np.expand_dims(data, axis=0)
    data = preprocess_input(data)

    if model == "Inception_V3":
        m = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3), pooling='max')
    elif model == "VGG_16":
        m = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='max')
    else:
        print("model not supported & wrong input, please check your param")
        return 0

    feature = m.predict(data)
    return feature


def main():
    if not os.path.exists("./dataset/Specgrams"):
        print("no data detected, exit.")
        exit()
    if not os.path.exists("./dataset/Transferred"):
        os.makedirs("./dataset/Transferred")
    if not os.path.exists("./dataset/Transferred/icp"):
        os.makedirs("./dataset/Transferred/icp")
    if not os.path.exists("./dataset/Transferred/icp/0"):
        os.makedirs("./dataset/Transferred/icp/0")
    if not os.path.exists("./dataset/Transferred/icp/1"):
        os.makedirs("./dataset/Transferred/icp/1")
    if not os.path.exists("./dataset/Transferred/icp/-1"):
        os.makedirs("./dataset/Transferred/icp/-1")
    if not os.path.exists("./dataset/Transferred/vgg"):
        os.makedirs("./dataset/Transferred/vgg")
    if not os.path.exists("./dataset/Transferred/vgg/0"):
        os.makedirs("./dataset/Transferred/vgg/0")
    if not os.path.exists("./dataset/Transferred/vgg/1"):
        os.makedirs("./dataset/Transferred/vgg/1")
    if not os.path.exists("./dataset/Transferred/vgg/-1"):
        os.makedirs("./dataset/Transferred/vgg/-1")
    dirs = "./dataset/Specgrams"

    if X == 1:
        x = "Inception_V3"
    else:
        x = "VGG_16"

    if x == "Inception_V3":
        m = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3), pooling='max')
    elif x == "VGG_16":
        m = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='max')
    else:
        m = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3), pooling='max')
    # to save RAM, please create one model.

    # get features...
    for i in range(-1, 2):
        for val in os.listdir("{}/{}".format(dirs, i)):
            if ".png" in val:
                if x == "Inception_V3":
                    img = image.load_img("{}/{}/{}".format(dirs, i, val),
                                         target_size=(299, 299),
                                         interpolation="bilinear")
                else:
                    img = image.load_img("{}/{}/{}".format(dirs, i, val),
                                         target_size=(224, 224),
                                         interpolation="bilinear")
                data = image.img_to_array(img)
                data = np.expand_dims(data, axis=0)
                data = preprocess_input(data)

                feature = m.predict(data)
                # save result
                if x == "Inception_V3":
                    np.save("./dataset/Transferred/icp/{}/{}".format(i, val[:-4]), feature)
                elif x == "VGG_16":
                    np.save("./dataset/Transferred/vgg/{}/{}".format(i, val[:-4]), feature)
    print("Function finished.")


if __name__ == "__main__":
    main()
