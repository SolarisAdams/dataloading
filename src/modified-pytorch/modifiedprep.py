import cv2
import torch
import numpy as np
import random

import os
import mmap
import sys

def readImageWithMmap(path):
    fd =  os.open(path, os.O_RDONLY | os.O_DIRECT) 
    mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ) 
    os.close(fd)
    img = np.fromstring(mm.read(), dtype="uint8")

    return img

def random_crop(image, crop_height, crop_width):
    max_x = image.shape[1] - crop_width + 1
    max_y = image.shape[0] - crop_height + 1
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)
    crop = image[y: y + crop_height, x: x + crop_width]
    return crop


def resize(img, square=224):
    height, width, _ = img.shape
    if height > width:
        height = height * square / width
        width = square
    else:
        width = width * square / height
        height = square
    dim = (int(width), int(height))
    img = cv2.resize(img, dim)
    return img


def transform(image):
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = resize(image)
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