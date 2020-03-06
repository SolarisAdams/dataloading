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

def preprocess_for_train(image):
    image = cv2.resize(image, (224, 224))
    image = random_crop(image,224,224)


    if random.random()<0.5:
        image = cv2.flip(image, 1)

    image = image / 255.0
    image = image - (0.485, 0.456, 0.406)
    image = image / (0.229, 0.224, 0.225)
    image = image.transpose((2, 0, 1))
    return image

def transform_for_mxnet(image, label):
    # image = image.asnumpy()

    image = cv2.resize(image, (224, 224))
    image = random_crop(image,224,224)


    if random.random()<0.5:
        image = cv2.flip(image, 1)

    image = image / 255.0
    image = image - (0.485, 0.456, 0.406)
    image = image / (0.229, 0.224, 0.225)
    image = image.transpose((2, 0, 1))


    return (image, label)



# import cv2
# import numpy as np 
# import os
# import mmap
# import random
# import pyarrow

# def readImageWithMmap(path):
#     fd =  os.open(path, os.O_RDONLY | os.O_DIRECT) 
#     mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ) 
#     os.close(fd)
#     return mm.read()

# def readImageFromHDFS(path):
#     fs = pyarrow.hdfs.connect('n1')
#     with fs.open('/' + path, 'rb') as f:
#         out = f.read()
#     return out

# def readImage(path):
#     with open(path, "rb") as f:
#         out = f.read()
#     return out

# def resize(img, square=224):
#     height, width, _ = img.shape
#     if height > width:
#         height = height * square / width
#         width = square
#     else:
#         width = width * square / height
#         height = square
#     dim = (int(width), int(height))
#     img = cv2.resize(img, dim)
#     return img

# def random_crop(image, crop_height, crop_width):
#     max_x = image.shape[1] - crop_width + 1
#     max_y = image.shape[0] - crop_height + 1
#     x = np.random.randint(0, max_x)
#     y = np.random.randint(0, max_y)
#     crop = image[y: y + crop_height, x: x + crop_width]
#     return crop

# def process(path):
#     img = readImageFromHDFS(path)
#     img = np.fromstring(img, dtype="uint8")
#     img = cv2.imdecode(img, cv2.IMREAD_COLOR)
#     img = resize(img)
#     img = random_crop(img,224,224)
#     if random.random()<0.5:
#         img= cv2.flip(img, 1)
#     if random.random() < 0.5:
#         img = cv2.flip(img, 0)
#     img = img / 255.0
#     img = img - (0.485, 0.456, 0.406)
#     img = img / (0.229, 0.224, 0.225)
#     return img