import mxnet as mx
import mxnet.gluon.data as data
import prep
import time


class mydataset(data.vision.ImageFolderDataset):
        def __getitem__(self, idx):
                img = prep.readImageWithMmap(self.items[idx][0])
                label = self.items[idx][1]
                if self._transform is not None:
                        return self._transform(img, label)
                return img, label

dataset= mydataset("/data/pytorch-imagenet-data/train/", flag=1,transform=prep.transform_for_mxnet)

data_loader = data.DataLoader(dataset, batch_size = prep.batch_size, num_workers = prep.worker, shuffle=True, pin_memory=True)


count = 0
end = time.time()
with open("loading/mxnet-gluon-"+str(prep.sleep_time)+"-"+str(prep.worker)+".txt","w") as f:
        for X, y in data_loader:
                cost = time.time()-end
                end = time.time()
                if prep.sleep:
                        time.sleep(prep.sleep_time)
                print(cost, "\t", prep.batch_size/cost, sep="", file=f)
                count += 1
                if count == prep.limit:
                        break