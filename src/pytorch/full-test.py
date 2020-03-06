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
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

arch_list = ["alexnet", "resnet18", "squeezenet1_1", "shufflenet_v2_x1_0", "mnasnet0_5"]
batch_size_list = [1024, 1024, 1024, 1536, 1024]



mode = sys.argv[1]
arch = arch_list[int(sys.argv[2])]
num_worker = int(sys.argv[3])

batch_size = batch_size_list[int(sys.argv[2])]
print_freq = 5
limit = 200


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None): #__init__是初始化该类的一些基础参数
        self.transform = transform #变换
        img_path = "/data/pytorch-imagenet-data/train/n01440764/n01440764_10040.JPEG"
        self.img = Image.open(img_path)
        self.label = 1
        self.sample = self.img
        if self.transform:
            self.sample = self.transform(self.sample)#对样本进行变换
    
    def __len__(self):#返回整个数据集的大小
        return 10000
    
    def __getitem__(self,index):#根据索引index返回dataset[index]

        return self.sample, self.label #返回该样本


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch, name, mode):
    batch_time = AverageMeter()
    data_time = AverageMeter()


    # switch to train mode
    model.train()
    # torch.cuda.synchronize()
    end = time.time()
    if mode == "train":
        for i, (data, label) in enumerate(train_loader):
            input_data = data
            target_data = label
            break
            # measure data loading time
        for i in range(limit):    
            input = input_data
            target = target_data
            end2 = time.time()
            load_time = end2-end
            data_time.update(end2 - end)
            
            target = target.cuda(async=True)
            end3 = time.time()

            trans_time = end3-end2

        

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)


            output = model(input_var)

            loss = criterion(output, target_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_time = time.time() - end3
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time))

            # with open("/home/Adama/dataloading/"+name+"-"+str(len(target))+".txt", "a+") as f:

            #     print(str(batch_time.val) + "\t" + str(len(input)/batch_time.val) + "\t" + str(time.asctime(time.localtime(time.time()))), file=f)
            
            with open("/home/Adama/dataloading/test_result/"+arch+"-"+mode+"-"+str(num_worker)+"-"+str(batch_size)+".txt", "a+") as f:
                print(str(batch_time.val) + "\t" + str(len(input)/batch_time.val) + "\t" + str(load_time) + "\t" + str(trans_time) + "\t" + str(train_time), file=f)
    else:
        for i, (input, target) in enumerate(train_loader):
            if i>=limit:
                break
            # print(input, target)
            # measure data loading time
            
            end2 = time.time()
            load_time = end2-end
            data_time.update(end2 - end)
            if mode != "load":
                target = target.cuda(async=True)
                end3 = time.time()

                trans_time = end3-end2

            

                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)

                # compute output
                output = model(input_var)
                # for googlenet case, there has three attributes
                #    logits, aux_logits2, aux_logits1
                loss = criterion(output, target_var)

                # measure accuracy and record loss

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # measure elapsed time
                # torch.cuda.synchronize()
                train_time = time.time() - end3
            else:
                train_time = 0
                trans_time = 0
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time))

            # with open("/home/Adama/dataloading/"+name+"-"+str(len(target))+".txt", "a+") as f:

            #     print(str(batch_time.val) + "\t" + str(len(input)/batch_time.val) + "\t" + str(time.asctime(time.localtime(time.time()))), file=f)
            
            with open("/home/Adama/dataloading/test_result/"+arch+"-"+mode+"-"+str(num_worker)+"-"+str(batch_size)+".txt", "a+") as f:
                print(str(batch_time.val) + "\t" + str(len(input)/batch_time.val) + "\t" + str(load_time) + "\t" + str(trans_time) + "\t" + str(train_time), file=f)


traindir = "/data/pytorch-imagenet-data/train/"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

if mode == "train":
    train_loader = torch.utils.data.DataLoader(
        FakeDataset(transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=True,
        num_workers=num_worker, pin_memory=True)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=True,
        num_workers=num_worker, pin_memory=True)




print("=> creating model '{}'".format(arch))
model = models.__dict__[arch]()

if arch.startswith('alexnet') or arch.startswith('vgg'):
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
else:
    model = torch.nn.DataParallel(model).cuda()

criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.SGD(model.parameters(), 0.1,
                            momentum=0.9,
                            weight_decay=1e-4)

cudnn.benchmark = True


# train for one epoch
with open("/home/Adama/dataloading/test_result/"+arch+"-"+mode+"-"+str(num_worker)+"-"+str(batch_size)+".txt", "w") as f:
    _=1
train(train_loader, model, criterion, optimizer, 0, arch, mode)