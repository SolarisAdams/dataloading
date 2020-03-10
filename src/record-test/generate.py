import tfrecord
import time
import random
def get_metadata(metadata_path='/home/Adama/dataloading/metadata'):
    with open(metadata_path, "r") as f:
        lines = f.readlines()
    records = []
    for line in lines:
        line = line.split()
        records.append((line[0] , int(line[1])))
    return records

def read_all(records, index, writer):

    for i in range(len(records)):
        with open(records[i][0], 'rb') as f:
            image_bytes = f.read()
        # print(type(records[i][1]))
        writer.write({
            "image": (image_bytes, "byte"),
            "label": ([records[i][1]], "int"),
            # "index": ([index[i]], "int")
        })
    return


start = time.time()
recordsall = get_metadata('/home/Adama/dataloading/metadata')
for i in range(4):
    records = recordsall[i*50000:i*50000+50000]
    index = [i for i in range(len(records))]

    random.shuffle(records)
    random.shuffle(index)



    writer = tfrecord.TFRecordWriter("/data/ImageNet/train-"+str(i)+".tfrecord")

    read_all(records,index, writer)

    writer.close()
    print(time.time()-start)