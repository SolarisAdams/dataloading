
import torchvision
import torch
import time
import numpy as np
from PIL import Image

sleep = 0
sleep_time = 0
batch_size = 1024
limit = 200
worker = 16




def fakeloader(path):
    return [0]


traindir = "/data/pytorch-imagenet-data/train/"


train_dataset = torchvision.datasets.ImageFolder(traindir, loader=fakeloader)


train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=worker, pin_memory=True)

count = 0
end = time.time()
with open("loading/pytorch-loader-"+str(worker)+"-"+str(batch_size)+".txt","w") as f:
        for i, (input, target) in enumerate(train_loader):
                # print(i, batch_size)
                cost = time.time()-end
                end = time.time()
                print(cost, "\t", batch_size/cost, sep="", file=f)
                count += 1
                if count == limit:
                        break