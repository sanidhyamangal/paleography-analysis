# import path for path functions
from path import Path

# import tensorflow
import tensorflow as tf

# import random
import random

class LoadData(object):
    """
    A data loader class for loading images from the respective dirs
    """

    # init function
    def __init__(self, path):

        # load root path
        self.path_to_dir = Path(path)

    def __load_labels(self):

        # path to all the images in list of str
        self.__all_images_path = [str(path) for path in self.path_to_dir.glob('*/*')]

        # shuffle the images to add variance
        random.shuffle(self.__all_images_path)

        # get the list of all the dirs
        all_root_labels = [str(path.name) for path in self.path_to_dir.glob('*') if path.isdir()]

        # design the dict of the labels
        self.root_labels = dict((c,i) for i, c in enumerate(all_root_labels))

        # add the labels for all the images
        all_images_labels = [self.root_labels[Path(image).parent.name] for image in self.__all_images_path]

        return all_images_labels

    # function to make process the images
    def __process_image(self,image_path):
        # read image into a raw format
        raw_image = tf.io.read_file(image_path)
        # decode the image
        decode_image = tf.image.decode_jpeg(raw_image, channels=3)

        # return the resized images
        return tf.image.resize(decode_image, [64, 64]) / 255.0

    # make a dataset emitter function
    def emit_dataset(self):
        # make a dataset for the labels 
        labels_dataset = tf.data.Dataset.from_tensor_slices(self.__load_labels())
        
        # develop an image dataset
        image_dataset = tf.data.Dataset.from_tensor_slices(self.__all_images_path)

        # process the image dataset
        image_dataset = image_dataset.map(self.__process_image)

        # combine and zip the dataset
        dataset = tf.data.Dataset.zip((image_dataset, labels_dataset))

        # shuffle and batch the dataset
        dataset = dataset.shuffle(1000).batch(64, drop_remainder=True)

        return dataset