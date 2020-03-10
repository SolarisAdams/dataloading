import multiprocessing
import sys
import modifiedprep as prep
import time


# NUM_WORKER = 32
# print(NUM_WORKER)

READ_WORKER = 16
TRANSFORM_WORKER = 16
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



def read_process(q_in, pipe):
    # count = 0
    while True:
        # count += 1
        deq = q_in.get()
        if deq is None:
            pipe.send(None)
            # print("read end:", count)
            break
        path, label = deq
        image = prep.readImageWithMmap(path)
        pipe.send((image, label))


def transform_process(pipe):
    # tot = 0.0
    # count = 0
    while True:
        # count += 1
        # start = time.time()
        deq = pipe.recv()
        # tot += time.time()-start
        if deq is None:
            # print("transform end", tot, count, tot/count)
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
    image_pipe = [multiprocessing.Pipe(False) for i in range(READ_WORKER)]

    readprocess = [ multiprocessing.Process(target=read_process, args=(index_queue,image_pipe[i][1])) for i in range(READ_WORKER) ]
    transformprocess = [ multiprocessing.Process(target=transform_process, args=(image_pipe[i][0], )) for i in range(TRANSFORM_WORKER) ]

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

    for p in transformprocess:
        p.join()      
    
    print("transform:\t", time.time()-start)


if __name__ == "__main__":
    import time
    #you should prepare your metadata 
    #and give your metadata path
    records = get_metadata('/home/Adama/dataloading/metadata')
    base = 200000
    records = records[base:base+100000]
    print(len(records))
    begin = time.time()
    main(records)
    end = time.time()
    print("speed:", len(records) / (end - begin))