import sys
import os
import torchvision.transforms as transforms
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import time
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

arch_list = ["alexnet", "resnet18", "squeezenet1_1", "shufflenet_v2_x1_0", "mnasnet0_5"]
batch_size_list = [1024, 1024, 1024, 1536, 1024]



mode = "load"
arch = arch_list[1]
num_worker = 16

batch_size = 1024
print_freq = 5
limit = 100

img_path = "/data/pytorch-imagenet-data/train/n01440764/n01440764_10040.JPEG"
fakeimage = Image.open(img_path)
fakeimage = fakeimage.convert('RGB')



def fakeloader(path):
    return fakeimage


traindir = "/data/pytorch-imagenet-data/train/"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])


train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),

            transforms.ToTensor(),

        ]), loader=fakeloader),
    batch_size=batch_size, shuffle=True,
    num_workers=num_worker, pin_memory=True)





# train for one epoch
with open("/home/Adama/dataloading/loading/"+arch+"-"+mode+"-"+str(num_worker)+"-"+str(batch_size)+".txt", "w") as f:
    _=1

epoch = 0


# switch to train mode

# torch.cuda.synchronize()
end = time.time()
for i, (input, target) in enumerate(train_loader):
    if i>=limit:
        break
    # print(input, target)
    # measure data loading time

    batch_time= time.time() - end
    end = time.time()

    # with open("/home/Adama/dataloading/"+name+"-"+str(len(target))+".txt", "a+") as f:

    #     print(str(batch_time.val) + "\t" + str(len(input)/batch_time.val) + "\t" + str(time.asctime(time.localtime(time.time()))), file=f)
    
    with open("/home/Adama/dataloading/loading/"+arch+"-"+mode+"-"+str(num_worker)+"-"+str(batch_size)+".txt", "a+") as f:
        print(str(batch_time) + "\t" + str(len(input)/batch_time), file=f)
