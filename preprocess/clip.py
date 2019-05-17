import os 
import cv2
import numpy as np
from functools import reduce


def main():
    # switch working folder to main path
    wf = os.getcwd()
    # print(wf)
    if "preprocess" in wf:
        os.chdir("../")
    # print(os.getcwd())

    if not os.path.exists("./dataset/Specgrams-unclipped"):
        print("specgrams not ready.")
        exit()

    dirs = "./dataset"

    for val in range(-1, 2):
        for i in os.listdir("{}/Specgrams-unclipped/{}/".format(dirs, val)):
            if ".bmp" in i:
                img = cv2.imread("{}/Specgrams-unclipped/{}/{}".format(dirs, val, i))

                dellist = []
                for k in range(img.shape[0]):
                    if list(reduce(lambda x, y: x & y, img[k, :, :])) == [255, 255, 255]:
                        dellist.append(k)
                img = np.delete(img, dellist, axis=0)

                dellist = []
                for k in range(img.shape[1]):
                    if list(reduce(lambda x, y: x & y, img[:, k, :])) == [255, 255, 255]:
                        dellist.append(k)
                img = np.delete(img, dellist, axis=1)

                # saving clipped fig......
                cv2.imwrite("{}/Specgrams/{}/{}".format(dirs, val, i), img)


if __name__ == "__main__":
    main()
