import prep
import torchvision
import torch
import time



traindir = "/data/pytorch-imagenet-data/train/"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
])

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

train_dataset = torchvision.datasets.ImageFolder(traindir, prep.transform, loader=fakeloader)


train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=prep.batch_size, shuffle=True,
        num_workers=prep.worker, pin_memory=True)

count = 0
end = time.time()
with open("loading/pytorch-realloader-"+str(prep.worker)+".txt","w") as f:
        for i, (input, target) in enumerate(train_loader):
                cost = time.time()-end
                end = time.time()
                if prep.sleep:
                        time.sleep(prep.sleep_time)
                print(cost, "\t", prep.batch_size/cost, sep="", file=f)
                count += 1
                if count == prep.limit:
                        break

