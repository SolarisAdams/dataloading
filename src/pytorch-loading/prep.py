import cv2
import torch
import numpy as np
import random
import mxnet as mx
import os
import mmap
import sys

print(sys.argv)

timelist = [0,0.1,0.2,0.3,0.4,0.5,0.8,1,2]

if len(sys.argv)>1:
    sleep = 1
    sleep_time = timelist[int(sys.argv[1])]
    batch_size = 512
    limit = 80
    worker = int(sys.argv[2])
    if worker == 1:
        limit = 50
else:
    sleep = 0
    sleep_time = 0
    batch_size = 1024
    limit = 200
    worker = 16


med = np.array([[[0.485, 0.456, 0.406]for _ in range(224)]for __ in range(224)])


def readImageWithMmap(path):
    fd =  os.open(path, os.O_RDONLY | os.O_DIRECT) 
    mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ) 
    os.close(fd)
    img = np.fromstring(mm.read(), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img


def CVLoader(path):
    image = cv2.imread(path, 1)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def random_crop(image, crop_height, crop_width):
    max_x = image.shape[1] - crop_width + 1
    max_y = image.shape[0] - crop_height + 1
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)
    crop = image[y: y + crop_height, x: x + crop_width]
    return crop


def transform(image):

    image = cv2.resize(image, (224, 224))
    image = random_crop(image,224,224)


    if random.random()<0.5:
        image = cv2.flip(image, 1)

    image = image / 255.0
    image = image - (0.485, 0.456, 0.406)
    image = image / (0.229, 0.224, 0.225)
    image = image.transpose((2, 0, 1))
    # print(image.shape)
    image = torch.from_numpy(image)
    return image

fakeimage = np.zeros((3,224,224))

def faketransform(image):
    return image

def fakeloader(path):
    
    return fakeimage