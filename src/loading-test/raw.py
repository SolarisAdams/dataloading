import multiprocessing
import sys
import prep
# import time
NUM_WORKER = 16
print(NUM_WORKER)

def get_metadata(metadata_path):
    with open("metadata", "r") as f:
        lines = f.readlines()
    records = []
    for line in lines:
        line = line.split()
        records.append((line[0] , int(line[1])))
    return records

def work_process(q_in):
    # tot = 0.0
    while True:
        # start = time.time()
        deq = q_in.get()
        # tot += time.time()-start
        if deq is None:
            # print(tot)
            break
        path, label = deq
        image = prep.readImageWithMmap(path)
        image = prep.transform(image)
        #process.process(path)
        #insert your process function


def main(records):
    if NUM_WORKER > 0:
        q_in = [ multiprocessing.Queue(1024) for i in range(NUM_WORKER) ]
        multiprocess = [ multiprocessing.Process(target=work_process, args=(q_in[i],)) for i in range(NUM_WORKER) ]
    
        for p in multiprocess:
            p.start()

        for i, record in enumerate(records):
            q_in[i % NUM_WORKER].put(record)

        for q in q_in:
            q.put(None)

        for p in multiprocess:
            p.join()

if __name__ == "__main__":
    import time
    #you should prepare your metadata 
    #and give your metadata path
    records = get_metadata('/home/Adama/dataloading/metadata')
    records = records[10000:12000]
    print(len(records))
    begin = time.time()
    main(records)
    end = time.time()
    print("speed:", len(records) / (end - begin))