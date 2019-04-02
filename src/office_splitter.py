import os
import sys

import numpy as np

def getRandomNumber():
    return np.random.rand()


def build_train_test_split(root_path, data_domain):
    source_path = root_path + "/" + data_domain + "/images/"
    train_split, test_split = [], []
    label = 0

    for sub_folder in os.listdir(source_path):
        if not sub_folder.startswith("."):
            sub_path = source_path + sub_folder + "/"
            count = 0

            for each_pic in os.listdir(sub_path):
                if each_pic.endswith(".jpg"):
                    count += 1
            train_count, train_threshold = 0, int(count * 0.8)
            test_count, test_threshold = 0, count - train_threshold

            for each_pic in os.listdir(sub_path):
                if each_pic.endswith(".jpg"):
                    if getRandomNumber() <= 0.8:
                        if train_count < train_threshold:
                            train_split.append((sub_path + each_pic, label))
                            train_count += 1
                        else:
                            test_split.append((sub_path + each_pic, label))
                            test_count += 1
                    else:
                        if test_count < test_threshold:
                            test_split.append((sub_path + each_pic, label))
                            test_count += 1
                        else:
                            train_split.append((sub_path + each_pic, label))
                            train_count += 1
            label += 1

    with open(root_path + "/train/" + data_domain + ".csv", "w") as train_f:
        last_flag = len(train_split) - 1
        for pair_index in range(len(train_split)):
            cpair = train_split[pair_index]
            train_f.writelines(str(cpair[0]) + "," + str(cpair[1]))
            if pair_index != last_flag:
                train_f.writelines("\n")

    with open(root_path + "/test/" + data_domain + ".csv", "w") as test_f:
        last_flag = len(test_split) - 1
        for pair_index in range(len(test_split)):
            cpair = test_split[pair_index]
            test_f.writelines(str(cpair[0]) + "," + str(cpair[1]))
            if pair_index != last_flag:
                test_f.writelines("\n")


np.set_printoptions(threshold=sys.maxsize)
root_path = os.getcwd()
root_path += "/../data/office"

amazon_path = "/amazon/images/"

build_train_test_split(root_path, "dslr")

