
import torchvision.transforms as transforms
from PIL import Image
import time


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')




NUM_WORKER = 1
print(NUM_WORKER)



def get_metadata(metadata_path='/home/Adama/metadata'):
    with open("metadata", "r") as f:
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
    # records = records[0:10000]
    print(len(records))
    begin = time.time()
    t=0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    for (path,label) in records:
        if t==0:
            image = pil_loader(path)
            t = 1
        # image = pil_loader(path)
        image2 = transform(image)
    end = time.time()
    print("speed:", len(records) / (end - begin))