import tfrecord

import PIL
from tfrecord.torch.dataset import TFRecordDataset

from io import BytesIO
import numpy as np
import torch.utils.data
from tfrecord import reader
from tfrecord import iterator_utils


class RecordDataset(TFRecordDataset):
    def __init__(self, data_path, index_path, description, transform=None, shuffle_queue_size=False):
        # super().__init__()
        self.data_path = data_path
        self.index_path = index_path
        self.description = description
        self.shuffle_queue_size = shuffle_queue_size
        self.transform = transform

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shard = worker_info.id, worker_info.num_workers
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        else:
            shard = None
        it = tfrecord.tfrecord_loader(
            self.data_path, self.index_path, self.description, shard)

        if self.shuffle_queue_size:
            it = iterator_utils.shuffle_iterator(it, self.shuffle_queue_size)

        def transformedit(it):
            for item in it:
                # print(item)
                image = self.transform(PIL.Image.open(BytesIO(item["image"])).convert("RGB"))
                label = item["label"][0]
                # print(image.shape, label)
                yield image, label
        it2 = transformedit(it)


        # print(it)    
        # image = self.transform(PIL.Image.open(BytesIO(it["image"])))
        # label = it["label"]
        return it2



# loader = tfrecord.tfrecord_loader("/data/ImageNet/train.tfrecord", "/data/ImageNet/train.idx", {
#     "image": "byte",
#     "label": "int",

# })

# def recordloader():
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         return img.convert('RGB')