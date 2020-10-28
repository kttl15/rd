import os
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


# * set gpu memory usage
device = tf.config.list_physical_devices("GPU")[0]
tf.config.experimental.set_memory_growth(device, True)


class Model:
    def __init__(self, weights_path: str):
        self.model = tf.keras.models.load_model(weights_path)
        self.model.summary()

    def predict(self, img_batch):
        predictions = self.model.predict_on_batch(img_batch).flatten()
        predictions = tf.nn.sigmoid(predictions)
        predictions = tf.where(predictions < 0.5, 0, 1)
        predictions = predictions.numpy()
        return predictions


BATCH_SIZE = 32
IMG_SIZE = (224, 224)
PATH = "data/"
class_names = pickle.load(open("class_names.pickle", "rb"))

# * load test images
validation_dir = os.path.join(PATH, "test")
val = image_dataset_from_directory(
    validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
)
val_batches = tf.data.experimental.cardinality(val)
test = val.take(val_batches // 5)

# * prefetching
AUTOTUNE = tf.data.experimental.AUTOTUNE
test = test.prefetch(buffer_size=AUTOTUNE)
image_batch, label_batch = test.as_numpy_iterator().next()

model = Model(weights_path="ckpt/weights_efficientnet_1603853937.1206157.hdf5")
predictions = model.predict(image_batch)

print("Predictions:\n", predictions)
print("Labels:\n", label_batch)


for i in range(9):
    print(class_names[predictions[i]])

