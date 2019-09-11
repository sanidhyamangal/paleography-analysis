# import path for path functions
from path import Path
# import tensorflow
import tensorflow as tf

# load the root path
root_path = Path('Paleographers_data\dataset_icr\dataset')

# convert the root path into a list of strings
all_images_path = [str(path) for path in root_path.glob('*\*')]

# get the list of all the dirs
all_root_labels = [str(path.name) for path in root_path.glob('*') if path.isdir()]

# convert the lables into a dict
root_labels = dict((c,i) for i,c in enumerate(all_root_labels))

# extract all the classes for all the images
all_images_labels = [root_labels[Path(image).parent.name] for image in all_images_path]
