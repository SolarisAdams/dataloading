import pathlib
import random
CMYK_IMAGES = [
    'n01739381_1309.JPEG',
    'n02077923_14822.JPEG',
    'n02447366_23489.JPEG',
    'n02492035_15739.JPEG',
    'n02747177_10752.JPEG',
    'n03018349_4028.JPEG',
    'n03062245_4620.JPEG',
    'n03347037_9675.JPEG',
    'n03467068_12171.JPEG',
    'n03529860_11437.JPEG',
    'n03544143_17228.JPEG',
    'n03633091_5218.JPEG',
    'n03710637_5125.JPEG',
    'n03961711_5286.JPEG',
    'n04033995_2932.JPEG',
    'n04258138_17003.JPEG',
    'n04264628_27969.JPEG',
    'n04336792_7448.JPEG',
    'n04371774_5854.JPEG',
    'n04596742_4225.JPEG',
    'n07583066_647.JPEG',
    'n13037406_4650.JPEG',
]

PNG_IMAGES = ['n02105855_2933.JPEG']

def get_paths(data_root):
    all_image_paths = list(data_root.glob("*/*"))
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    random.shuffle(all_image_paths)
    with open("metadata", "w") as f:
        for path in all_image_paths:
            label = label_to_index[path.parent.name]
            f.writelines(str(path) + '\t' + str(label) + '\n')

root = pathlib.Path("/data/pytorch-imagenet-data/train")
print(root)
get_paths(root)

# import tensorflow as tf 
# import random
# import pathlib

# dataset_dir = "/data/pytorch-imagenet-data/"
# train_dir = dataset_dir + "train"
# NUM_IMAGES = 1281144

# CMYK_IMAGES = [
#     'n01739381_1309.JPEG',
#     'n02077923_14822.JPEG',
#     'n02447366_23489.JPEG',
#     'n02492035_15739.JPEG',
#     'n02747177_10752.JPEG',
#     'n03018349_4028.JPEG',
#     'n03062245_4620.JPEG',
#     'n03347037_9675.JPEG',
#     'n03467068_12171.JPEG',
#     'n03529860_11437.JPEG',
#     'n03544143_17228.JPEG',
#     'n03633091_5218.JPEG',
#     'n03710637_5125.JPEG',
#     'n03961711_5286.JPEG',
#     'n04033995_2932.JPEG',
#     'n04258138_17003.JPEG',
#     'n04264628_27969.JPEG',
#     'n04336792_7448.JPEG',
#     'n04371774_5854.JPEG',
#     'n04596742_4225.JPEG',
#     'n07583066_647.JPEG',
#     'n13037406_4650.JPEG',
# ]

# PNG_IMAGES = ['n02105855_2933.JPEG']

# def get_paths(data_root):
#     all_image_paths = list(data_root.glob("*/*"))
#     all_image_paths = [str(path) for path in all_image_paths]
#     for path in all_image_paths:
#         file_name = path.split('/')[-1]
#         if file_name in CMYK_IMAGES or file_name in PNG_IMAGES:
#             all_image_paths.remove(path)
#             print("remove: " + path)
#     random.shuffle(all_image_paths)
#     print("IMAGE NUM", len(all_image_paths))
#     return all_image_paths

# def get_labels(data_root, all_image_paths):
# 	label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
# 	label_to_index = dict((name, index) for index, name in enumerate(label_names))
# 	all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
# 	return all_image_labels

# def prepare_metadata(file_path = train_dir, metadata_name = "metadata"):
#     file_path = pathlib.Path(file_path)
#     all_image_paths = get_paths(file_path)
#     all_image_labels = get_labels(file_path, all_image_paths)

#     image_dataset = tf.data.Dataset.from_tensor_slices(all_image_paths)
#     label_dataset = tf.data.Dataset.from_tensor_slices(all_image_labels)
#     dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
#     dataset = dataset.cache(metadata_name)

#     for data in dataset.as_numpy_iterator():
#         pass
#     return True

# def get_dataset(metadata_name = "metadata"):
#     #init dataset struct to read metadata correctly
#     dataset1 = tf.data.Dataset.from_tensor_slices(["1","2"]).repeat(NUM_IMAGES)
#     dataset2 = tf.data.Dataset.from_tensor_slices([1,2]).repeat(NUM_IMAGES)
#     dataset = tf.data.Dataset.zip((dataset1, dataset2)).cache(metadata_name)
#     return dataset
