import multiprocessing
import sys
import modifiedprep as prep
import time


# NUM_WORKER = 32
# print(NUM_WORKER)

READ_WORKER = 1
TRANSFORM_WORKER = 1
print("read worker:\t", READ_WORKER)
print("transform worker:\t", TRANSFORM_WORKER)

def get_metadata(metadata_path):
    with open("metadata", "r") as f:
        lines = f.readlines()
    records = []
    for line in lines:
        line = line.split()
        records.append((line[0] , int(line[1])))
    return records



def read_process(q_in, q_out):
    # count = 0
    while True:
        # count += 1
        deq = q_in.get()
        if deq is None:
            # print("read end:", count)
            break
        path, label = deq
        image = prep.readImageWithMmap(path)
        q_out.put((image, label))


def transform_process(q_in):
    tot = 0.0
    count = 0
    while True:
        count += 1
        start = time.time()
        deq = q_in.get()
        tot += time.time()-start
        if deq is None:
            print("transform end", tot, count, tot/count)
            break
        image, label = deq
        image = prep.transform(image)






# def work_process(q_in):
#     while True:
#         deq = q_in.get()
#         if deq is None:
#             break
#         path, label = deq
#         image = prep.readImageWithMmap(path)
#         # image = prep.transform(image)
#         #process.process(path)
#         #insert your process function


def main(records):

    index_queue = multiprocessing.Queue(2048)
    image_queue = [multiprocessing.Queue(16) for i in range(READ_WORKER)]

    readprocess = [ multiprocessing.Process(target=read_process, args=(index_queue,image_queue[i])) for i in range(READ_WORKER) ]
    transformprocess = [ multiprocessing.Process(target=transform_process, args=(image_queue[i % READ_WORKER], )) for i in range(TRANSFORM_WORKER) ]

    for p in readprocess:
        p.start()
    for p in transformprocess:
        p.start()

    start = time.time()
    for i, record in enumerate(records):
        index_queue.put(record)
    print("put:\t", time.time()-start)

    for _ in range(READ_WORKER):
        index_queue.put(None)

    for p in readprocess:
        p.join()
    end = time.time()
    print("read:\t", end-start)

    for i in range(TRANSFORM_WORKER):
        image_queue[i%READ_WORKER].put(None)

    for p in transformprocess:
        p.join()      
    
    print("transform:\t", time.time()-start)


if __name__ == "__main__":
    import time
    #you should prepare your metadata 
    #and give your metadata path
    records = get_metadata('/home/Adama/dataloading/metadata')
    base = 100000
    records = records[base:base+2000]
    print(len(records))
    begin = time.time()
    main(records)
    end = time.time()
    print("speed:", len(records) / (end - begin))