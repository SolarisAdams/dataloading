import cProfile
import numpy as np
import random
import mxnet as mx
import os
import mmap
import cv2
import torch
import time
path_list = []
with open("metadata","r") as f:
    for line in f:
        line = line.split()
        path_list.append(line[0])


def random_crop(image, crop_height, crop_width):
    max_x = image.shape[1] - crop_width + 1
    max_y = image.shape[0] - crop_height + 1
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)
    crop = image[y: y + crop_height, x: x + crop_width]
    return crop


def normalize(image):
    image = image / 255.0
    image = image - (0.485, 0.456, 0.406)
    image = image / (0.229, 0.224, 0.225)
    return image
with open("loading/disk.ans","w") as f:

    for path in path_list:
        end = time.time()
        fd =  os.open(path, os.O_RDONLY | os.O_DIRECT) 
        mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ) 
        os.close(fd)
        data = mm.read()
        print(time.time()-end, file=f)


        img = np.fromstring(data, dtype="uint8")

        image = cv2.imdecode(img, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = random_crop(image,224,224)


        if random.random()<0.5:
            image = cv2.flip(image, 1)

        image = normalize(image)
        image = image.transpose((2, 0, 1))
        # print(image.shape)
        image = torch.from_numpy(image)
     






