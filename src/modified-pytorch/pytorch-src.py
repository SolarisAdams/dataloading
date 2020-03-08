import prep
import torchvision
import torch
import time
import dataloaderx


traindir = "/data/pytorch-imagenet-data/train/"


train_dataset = torchvision.datasets.ImageFolder(traindir, prep.transform, loader=prep.readImageWithMmap)


train_loader = dataloaderx.mydataloader(
        train_dataset,
        batch_size=prep.batch_size//2, shuffle=True,
        num_workers=prep.worker, pin_memory=True)

count = 0
end = time.time()
with open("loading/pytorch-"+str(prep.sleep_time)+"-"+str(prep.worker)+".txt","w") as f:
        for i, (input, target) in enumerate(train_loader):
                print(input[0].shape, input[1].shape)
                cost = time.time()-end
                end = time.time()
                if prep.sleep:
                        time.sleep(prep.sleep_time)
                print(cost, "\t", prep.batch_size/cost, sep="", file=f)
                count += 1
                if count == prep.limit:
                        break

