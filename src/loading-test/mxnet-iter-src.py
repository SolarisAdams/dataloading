import mxnet as mx
import prep
import time


data_iter = mx.image.ImageIter(batch_size=prep.batch_size, data_shape=(3, 224, 224),
                            label_width = 1,
                            path_imglist="/home/Adama/dataloading/train.lst",
                            path_root="/data/pytorch-imagenet-data/train/",
                            aug_list = mx.image.CreateAugmenter((3,224,224),resize=224,rand_crop=True,\
                                            rand_mirror=True,mean=True,std=True))

# data_iter的类型是mxnet.image.ImageIter
#reset()函数的作用是：resents the iterator to the beginning of the data
print("start")
data_iter.reset()


print("start")
count = 0
end = time.time()
with open("loading/mxnet-iter.txt","w") as f:
        for batch in data_iter:
                cost = time.time()-end
                if prep.sleep:
                        time.sleep(prep.sleep_time)
                print(cost, "\t", prep.batch_size/cost, sep="", file=f)
                print(cost, "\t", prep.batch_size/cost, sep="")
                count += 1
                if count == prep.limit/2:
                        break
                end = time.time()



