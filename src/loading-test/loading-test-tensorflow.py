import tensorflow as tf 
import prep
import time


def get_metadata(metadata_name = "metadata"):
    with open(metadata_name, 'r') as f:
        out = f.readlines()
    paths = []
    labels = []
    for line in out:
        line = line.split()
        paths.append(line[0])
        # labels.append(line[1])
    return tf.data.Dataset.from_tensor_slices(paths)



def get_dataset(pro=True):

    def process(path):
        path = path.numpy().decode()
        image = prep.CVLoader(path)
        image = prep.preprocess_for_train(image)    
        return image
       
    def process_batch(inputs):
        # print(inputs)
        # print(type(inputs))

        result = []
        for path in inputs:
            result.append(process(path))
        return result
        
    if pro:
        dataset = get_metadata(metadata_name = "/home/Adama/dataloading/metadata").shuffle(1024).map(lambda x : tf.py_function(process, [x], [tf.float32]), num_parallel_calls=prep.worker).batch(prep.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = get_metadata(metadata_name = "/home/Adama/dataloading/metadata").shuffle(1024).batch(prep.batch_size).map(lambda x : tf.py_function(process_batch, [x], [tf.float32]), num_parallel_calls=prep.worker).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

#这里我准备了两个dataset 分别是 dataset(True) 和 dataset(false) 只是batch 和 map 的顺序不一样，希望都测下 num_worker from 1 to 16的情况

dataset = get_dataset(pro=False)

count = 0
end = time.time()
with open("loading/tensorflowFalse-"+str(prep.sleep_time)+"-"+str(prep.worker)+".txt","w") as f:
        for batch in dataset:
                cost = time.time()-end
                end = time.time()
                if prep.sleep:
                        time.sleep(prep.sleep_time)
                print(cost, "\t", prep.batch_size/cost, sep="", file=f)
                count += 1
                if count == prep.limit:
                        break


