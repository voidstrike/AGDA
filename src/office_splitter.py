import os
import sys
from shutil import copyfile

import numpy as np


def build_train_test_split(root_path, data_domain, tgt_train, tgt_test):
    source_path = root_path + data_domain + "/images/"
    tgt_train_path = tgt_train + data_domain + "/"
    tgt_test_path = tgt_test + data_domain + "/"

    if not os.path.exists(tgt_train_path):
        os.mkdir(tgt_train_path)
    if not os.path.exists(tgt_test_path):
        os.mkdir(tgt_test_path)

    for sub_folder in os.listdir(source_path):
        if not sub_folder.startswith("."):
            tmp_train_path = tgt_train_path + sub_folder + "/"
            tmp_test_path = tgt_test_path + sub_folder + "/"

            if not os.path.exists(tmp_train_path):
                os.mkdir(tmp_train_path)
            if not os.path.exists(tmp_test_path):
                os.mkdir(tmp_test_path)

            sub_path = source_path + sub_folder + "/"
            count = 0

            for each_pic in os.listdir(sub_path):
                if each_pic.endswith(".jpg"):
                    count += 1
            train_count, train_threshold = 0, int(count * 0.8)
            test_count, test_threshold = 0, count - train_threshold

            for each_pic in os.listdir(sub_path):
                if each_pic.endswith(".jpg"):
                    if np.random.rand() <= 0.8:
                        if train_count < train_threshold:
                            copyfile(sub_path + each_pic, tmp_train_path + each_pic)
                            train_count += 1
                        else:
                            copyfile(sub_path + each_pic, tmp_test_path + each_pic)
                            test_count += 1
                    else:
                        if test_count < test_threshold:
                            copyfile(sub_path + each_pic, tmp_test_path + each_pic)
                            test_count += 1
                        else:
                            copyfile(sub_path + each_pic, tmp_train_path + each_pic)
                            train_count += 1


np.set_printoptions(threshold=sys.maxsize)
root_path = os.getcwd()

root_path += "/../data/office/"
train_path = root_path + "/train/"
test_path = root_path + "/test/"

amazon_path = "/amazon/images/"

build_train_test_split(root_path, "webcam", train_path, test_path)

