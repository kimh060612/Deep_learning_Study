import numpy as np
import os
import cv2 

local_dir = os.path.expanduser('~') + "/Data/imagenet-mini"
list_train_data = os.listdir(local_dir + "/train/")
list_validation_data = os.listdir(local_dir + "/val/")

train_image = []
train_label = []
test_image = []
test_label = []

n_classes = 20
label = 0
cnt = 0
for i in range(n_classes):
    con = list_train_data[i]

    Image_dir = local_dir + "/train/" + con
    Image_list = os.listdir(Image_dir) 
    for img in Image_list:
        Img = cv2.imread(Image_dir + "/" + img)
        Img = cv2.resize(Img, (224, 224), interpolation=cv2.INTER_LINEAR)
        train_image.append(Img)
        train_label.append(label)

    test_Image_dir = local_dir + "/val/" + con
    test_Image_list = os.listdir(test_Image_dir)

    for img in test_Image_list:
        Img = cv2.imread(test_Image_dir + "/" + img)
        Img = cv2.resize(Img, (224, 224), interpolation=cv2.INTER_LINEAR)
        test_image.append(Img)
        test_label.append(label)

    label += 1

train_image = np.array(train_image, dtype="float32")
train_label = np.array(train_label, dtype="float32")
train_label = np.transpose(train_label)

test_image = np.array(test_image, dtype="float32")
test_label = np.array(test_label, dtype="float32")
test_label = np.transpose(test_label)

np.save("train_Image.npy", train_image)
np.save("train_label.npy", train_label)
np.save("test_Image.npy", test_image)
np.save("test_label.npy", test_label)

