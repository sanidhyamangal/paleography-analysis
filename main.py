# import tensorflow file
import tensorflow as tf

# dataloader lib
from load_imageds import LoadData

# load data obj for loading the objects
imageLoader = LoadData("Paleographers_data\dataset_icr\dataset")

# load the imageds
imageds = imageLoader.emit_dataset()

# develop the model
model = tf.keras.models.Sequential()

# first conv1 layer (1)
model.add(tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=[64, 64, 3]))

# max pool layer
model.add(tf.keras.layers.MaxPooling2D(2,2))

# a next conv2d  (2)
model.add(tf.keras.layers.Conv2D(64, (3,3), activation="relu"))

# a  max pool layer
model.add(tf.keras.layers.MaxPooling2D(2,2))

# a conv2d layer (3)
model.add(tf.keras.layers.Conv2D(64, (3,3), activation="relu"))

# a max pool layer
model.add(tf.keras.layers.MaxPooling2D(2,2))

# flatten all the layer
model.add(tf.keras.layers.Flatten())

# a dense layer (1)
model.add(tf.keras.layers.Dense(512, activation="relu"))

# drop out layer
model.add(tf.keras.layers.Dropout(rate=0.4))

# final dense layer for the classification
model.add(tf.keras.layers.Dense(len(imageLoader.root_labels.keys()), activation="softmax"))

# compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

print(model.summary())

# train the model
model.fit(imageds, epochs=5)