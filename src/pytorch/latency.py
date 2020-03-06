import os
import time

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy

traindir = "/data/pytorch-imagenet-data/train/"

total_batch_size = 1024
batch_size = 1024

with open("/home/Adama/dataloading/latency.ans", "w") as f:
    with open("/home/Adama/dataloading/latency.data", "w") as f2:
        for i in range(8):
            
            batch_num = total_batch_size/batch_size
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
            train_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(traindir, transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=batch_size, shuffle=True,
                num_workers=16, pin_memory=True)

            count = 0
            end2 = time.time()
            latency = []
            for i, (input, target) in enumerate(train_loader):
                if i % batch_num == 0:
                    print(time.time()-end2, file=f2)
                    latency.append(time.time()-end2)
                    end2 = time.time()
                    count += 1
                    if count == 10:
                        head = time.time()
                    if count >= 100:
                        break
            speed = 90*1024/(end2-head)
            std = numpy.std(latency, ddof=1)
            print(batch_size, speed, (end2-head)/90, std, file=f)
            print(batch_size, speed, (end2-head)/90, std)
            batch_size = int(batch_size / 2)

            



