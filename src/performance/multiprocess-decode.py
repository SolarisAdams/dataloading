import multiprocessing
import sys
import cv2
import random
import numpy as np
import os
import mmap
import torchvision.transforms as transforms
import simplejpeg
import torch
import time



def random_crop(image, crop_height, crop_width):
    max_x = image.shape[1] - crop_width + 1
    max_y = image.shape[0] - crop_height + 1
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)
    crop = image[y: y + crop_height, x: x + crop_width]
    return crop

# def pil_loader(path):
#     # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         return img.convert('RGB')

def readImageWithMmap(path):
    fd =  os.open(path, os.O_RDONLY | os.O_DIRECT) 
    mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ) 
    os.close(fd)
    img = np.fromstring(mm.read(), dtype="uint8")

    return img

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
    return image


if len(sys.argv)>1:
    NUM_WORKER = int(sys.argv[1])
else:
    NUM_WORKER = 1
print(NUM_WORKER)

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
# transform = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#     ])

def get_metadata(metadata_path='/home/Adama/metadata'):
    with open("metadata", "r") as f:
        lines = f.readlines()
    records = []
    for line in lines:
        line = line.split()
        records.append((line[0] , int(line[1])))
    return records

def work_process(q_in):
    t=0
    # mytransform = transform
    # start = time.time()
    while True:
        deq = q_in.get()
        if deq is None:
            # print(time.time()-start)
            break
        image = deq
        # if t==0:
        #     image = readImageWithMmap(path)
        #     t = 1
        # image = readImageWithMmap(path)
        # try:
        #     image = simplejpeg.decode_jpeg(image, colorspace="bgr")
        # except:
        #     image = np.asarray(bytearray(image), dtype="uint8")
        #     image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = np.asarray(bytearray(image), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        #process.process(path)
        #insert your process function

def read_all(records):
    result = [None] * len(records)
    for i in range(len(records)):
        with open(records[i][0], 'rb') as f:
            result[i] = f.read()
    return result

def main(records, raw):
    if NUM_WORKER > 0:
        
        q_in = [ multiprocessing.Queue(1024) for i in range(NUM_WORKER) ]
        multiprocess = [ multiprocessing.Process(target=work_process, args=(q_in[i],)) for i in range(NUM_WORKER) ]
    
        for p in multiprocess:
            p.start()

        for i, data in enumerate(raw):
            q_in[i % NUM_WORKER].put(data)

        for q in q_in:
            q.put(None)

        for p in multiprocess:
            p.join()

if __name__ == "__main__":

    #you should prepare your metadata 
    #and give your metadata path
    records = get_metadata('/home/Adama/dataloading/metadata')
    records = records[10000:20000+ 2000*NUM_WORKER]
    print(len(records))
    raw = read_all(records)
    begin = time.time()
    main(records,raw)
    end = time.time()
    print("speed:", len(records) / (end - begin))