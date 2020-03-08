import cv2
import random
import numpy as np
import os
import time


BATCH = 2

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

def transform(raw):
    # image = np.asarray(bytearray(raw), dtype="uint8")
    for image in raw:
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # image = resize(image)
        # image = random_crop(image,224,224)


        # if random.random()<0.5:
        #     image = cv2.flip(image, 1)

        # image = image / 255.0
        # image = image - (0.485, 0.456, 0.406)
        # image = image / (0.229, 0.224, 0.225)
        # image = image.transpose((2, 0, 1))
    return image

def read_all(records):
    result = [None] * len(records)
    for i in range(len(records)):
        with open(records[i][0], 'rb') as f:
            result[i] = np.asarray(bytearray(f.read()), dtype="uint8")
    return result
        


def get_metadata(metadata_path='/home/Adama/dataloading/metadata'):
    with open(metadata_path, "r") as f:
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
    records = records[0:20000]
    random.shuffle(records)
    raws = read_all(records)
    print(len(records))

    begin = time.time()
    for i in range(len(records)//BATCH):
        transform(raws[BATCH*i:BATCH*i+BATCH])
    end = time.time()
    print("speed:", len(records) / (end - begin))