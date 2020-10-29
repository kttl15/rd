import os
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import img_to_array, load_img


# * set gpu memory usage
device = tf.config.list_physical_devices("GPU")[0]
tf.config.experimental.set_memory_growth(device, True)


class Model:
    def __init__(self, weights_path: str):
        """Loads a "weights + model" model and prints the model summary

        Args:
            weights_path (str): path to weights
        """
        self.model = tf.keras.models.load_model(weights_path)
        self.model.summary()

    def predict(self, img_batch):
        """Returns a prediction of the image batch.

        Args:
            img_batch: Image batch.

        Returns:
            predictions (np.array()): a numpy array of predictions.
        """
        predictions = self.model.predict_on_batch(img_batch).flatten()
        predictions = tf.nn.sigmoid(predictions)
        predictions = predictions.numpy()
        pred = []

        for i in range(int(len(predictions) / 189)):
            pred.append(predictions[i * 189 : (i + 1) * 189])
        return pred


def select_top_3(predictions):
    class_names = pickle.load(open("class_names.pickle", "rb"))

    data = []
    for pred in predictions:
        top_3_labels = pred.argsort()[-5:][::-1]
        top_3_scores = [pred[i] for i in top_3_labels]
        data.append([{class_names[k]: v} for k, v in zip(top_3_labels, top_3_scores)])
    return data


BATCH_SIZE = 32
IMG_SIZE = (224, 224)
PATH = "frames"

# * load test images
test = image_dataset_from_directory(
    PATH, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
)

# * prefetching
AUTOTUNE = tf.data.experimental.AUTOTUNE
test = test.prefetch(buffer_size=AUTOTUNE)
image_batch, label_batch = test.as_numpy_iterator().next()

model = Model(weights_path="ckpt/weights_efficientnet_1603853937.1206157.hdf5")
predictions = model.predict(image_batch)

predictions = select_top_3(predictions)


print(predictions)

