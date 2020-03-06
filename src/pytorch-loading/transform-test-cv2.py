import cv2
import torch
import numpy as np
import random
import mxnet as mx
import os
import mmap
import sys
import torchvision.transforms as transforms
from PIL import Image
import time


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


def transform(image):
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
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


NUM_WORKER = 1
print(NUM_WORKER)



def get_metadata(metadata_path='/home/Adama/metadata'):
    with open("metadata", "r") as f:
        lines = f.readlines()
    records = []
    for line in lines:
        line = line.split()
        records.append((line[0] , int(line[1])))
    return records


if __name__ == "__main__":
    #you should prepare your metadata 
    #and give your metadata path
    records = get_metadata('/home/Adama/dataloading/metadata')
    records = records[0:10000]
    print(len(records))
    begin = time.time()
    t=0


    for (path,label) in records:
        # if t==0:
        #     image = readImageWithMmap(path)
        #     t = 1
        image = readImageWithMmap(path)
        # image2 = transform(image)
    end = time.time()
    print("speed:", len(records) / (end - begin))