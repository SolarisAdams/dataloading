import prep
import torchvision
import torch
import time
import torchvision.transforms as transforms


traindir = "/data/pytorch-imagenet-data/train/"

def fakeloader(path):
    return [0]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

train_dataset = torchvision.datasets.ImageFolder(traindir,transform=transform)



count = 0
end = time.time()
with open("loading/pytorch-realdataset-"+str(prep.worker)+".txt","w") as f:
    for i, (input, target) in enumerate(train_dataset):
        if i % prep.batch_size == 0:
            # print(input)
            cost = time.time()-end
            end = time.time()
            if prep.sleep:
                    time.sleep(prep.sleep_time)
            print(cost, "\t", prep.batch_size/cost, sep="", file=f)
            count += 1
            if count == prep.limit:
                    break