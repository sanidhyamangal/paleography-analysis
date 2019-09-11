# import path for path functions
from path import Path
# import tensorflow
import tensorflow as tf

# import random
import random


# load the root path
root_path = Path('Paleographers_data\dataset_icr\dataset')

# convert the root path into a list of strings
all_images_path = [str(path) for path in root_path.glob('*\*')]

# create a randomized images
random.shuffle(all_images_path)

# get the list of all the dirs
all_root_labels = [str(path.name) for path in root_path.glob('*') if path.isdir()]

# convert the lables into a dict
root_labels = dict((c,i) for i,c in enumerate(all_root_labels))

# extract all the classes for all the images
all_images_labels = [root_labels[Path(image).parent.name] for image in all_images_path]

# function to make process the images
def process_image(image_path):
    # read image into a raw format
    raw_image = tf.io.read_file(image_path)
    # decode the image
    decode_image = tf.image.decode_image(raw_image)

    return decode_image / 255

# make a dataset of all the labels
image_dataset = tf.data.Dataset.from_tensor_slices(all_images_path)

# final image dataset
image_dataset = image_dataset.map(process_image)

# make a dataset for the labels
labels_dataset = tf.data.Dataset.from_tensor_slices(all_images_labels)

# make a combined dataset
dataset = tf.data.Dataset.zip((image_dataset, labels_dataset))

# shuffle dataset
dataset = dataset.shuffle(1000).batch(64, drop_remainder=True)
