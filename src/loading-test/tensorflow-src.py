import tensorflow as tf 
import prep
import time
# import datasets_util


# NUM_IMAGES = 1281144


def get_dataset2(metadata_name = "metadata"):
    #init dataset struct to read metadata correctly
    with open("metadata", 'r') as f:
        out = f.readlines()
    paths = []
    labels = []
    for line in out:
        line = line.split()
        paths.append(line[0])
        labels.append(int(line[1]))
    dataset1 = tf.data.Dataset.from_tensor_slices(paths)
    dataset2 = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((dataset1, dataset2))
    return dataset


# def get_dataset2(metadata_name = "metadata"):
#     #init dataset struct to read metadata correctly
#     dataset1 = tf.data.Dataset.from_tensor_slices(["1","2"]).repeat(NUM_IMAGES)
#     dataset2 = tf.data.Dataset.from_tensor_slices([1,2]).repeat(NUM_IMAGES)
#     dataset = tf.data.Dataset.zip((dataset1, dataset2)).cache(metadata_name)
#     return dataset

def get_dataset(pro=True):

    def process(path, label):
        path = path.numpy().decode()
        # def get_path(path):
        #     return path
        # path = tf.py_function(get_path, [path], [tf.string])[0]


        image = prep.CVLoader(path)
        
        image = prep.preprocess_for_train(image)    
        image = tf.convert_to_tensor(image)
    
        return image, label
        #vgg_preprocessing_v2.preprocess_for_train is from a famous third party repo called tensorflow slim 


    # print("beign generate metadata")
    # if datasets_util.prepare_metadata(metadata_name = "/home/Adama/tensorflow/metadata") is True:
    #     print("generate metadata successfully")    
        
    dataset = get_dataset2(metadata_name = "/home/Adama/dataloading/metadata").shuffle(1024).map(lambda x, y: tf.py_function(process, [x, y], [tf.float64, tf.int32]), num_parallel_calls=prep.worker).batch(prep.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

dataset = get_dataset(pro=True)

count = 0
end = time.time()
with open("loading/tensorflow-"+str(prep.sleep_time)+"-"+str(prep.worker)+".txt","w") as f:
        for batch in dataset:
                cost = time.time()-end
                end = time.time()
                if prep.sleep:
                        time.sleep(prep.sleep_time)
                print(cost, "\t", prep.batch_size/cost, sep="", file=f)
                count += 1
                if count == prep.limit:
                        break
