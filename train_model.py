import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint


# * set gpu memory usage
device = tf.config.list_physical_devices("GPU")[0]
tf.config.experimental.set_memory_growth(device, True)

tf.get_logger().setLevel("INFO")


class Model:
    def __init__(self):
        self.base_model_name = None
        self.BATCH_SIZE = 32
        self.IMG_SIZE = (160, 160)
        self.IMG_SHAPE = IMG_SIZE + (3,)

        self.data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
                tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
            ]
        )

    def load_model(self, base_model_name: str):
        """[summary]
        
        Load/download a model.

        Args:
            base_model_name (str): ['efficientnet', 'resnet50', 'vgg16']

        Raises:
            ValueError: error if base model name not in list
        """
        if base_model_name not in ["efficientnet", "resnet50", "vgg16"]:
            raise ValueError(
                "base model name not one of ['efficientnet', 'resnet50', 'vgg16']"
            )

        self.base_model_name = base_model_name

        if base_model_name == "efficientnet":
            self.preprocessor = tf.keras.applications.efficientnet.preprocess_input
            self.base_model = tf.keras.applications.EfficientNetB7(
                include_top=False, input_shape=self.IMG_SHAPE
            )
        elif base_model_name == "resnet50":
            self.preprocessor = tf.keras.applications.resnet.preprocess_input
            self.base_model = tf.keras.applications.ResNet50V2(
                include_top=False, input_shape=self.IMG_SHAPE
            )
        elif base_model_name == "vgg16":
            self.preprocessor = tf.keras.applications.vgg16.preprocess_input
            self.base_model = tf.keras.applications.VGG16(
                include_top=False, input_shape=self.IMG_SHAPE
            )

        self.base_model.trainable = False

    def __is_model_set__(self):
        if not self.base_model_name:
            raise NameError("base model not set. Please load a base model first")

    def build_model(self, train_data, output_layers):
        self.__is_model_set__()
        self.img_batch, _ = next(iter(train_data))
        self.feature_batch = self.base_model(self.img_batch)
        inputs = tf.keras.layers.Input(shape=self.IMG_SHAPE)
        x = self.data_augmentation(inputs)
        x = self.preprocessor(x)
        x = self.base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = output_layers(x)
        model = tf.keras.Model(inputs, outputs)
        return model


# * read data
PATH = "data/"
train_dir = os.path.join(PATH, "train")
validation_dir = os.path.join(PATH, "test")

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train = image_dataset_from_directory(
    train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
)

val = image_dataset_from_directory(
    validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
)

# * create a test dataset from validation dataset
val_batches = tf.data.experimental.cardinality(val)
test = val.take(val_batches // 5)
val = val.skip(val_batches // 5)
print("Number of validation batches: %d" % tf.data.experimental.cardinality(val))
print("Number of test batches: %d" % tf.data.experimental.cardinality(test))


# * enables prefetching. Prepares next training batch while model is training on current batch.
AUTOTUNE = tf.data.experimental.AUTOTUNE
train = train.prefetch(buffer_size=AUTOTUNE)
val = val.prefetch(buffer_size=AUTOTUNE)
test = test.prefetch(buffer_size=AUTOTUNE)

# * init model
model = Model()
model.load_model("vgg16")

# * add output layer
output_layers = tf.keras.layers.Dense(189)
model = model.build_model(train, output_layers)
base_learning_rate = 0.001
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# * callbacks
tb = TensorBoard(log_dir="logs", update_freq="epoch", write_graph=True)

red_lr = ReduceLROnPlateau()

ckpt = ModelCheckpoint(
    filepath="ckpt/weights.{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.hdf5",
    save_weights_only=False,
    monitor="val_accuracy",
    save_best_only=True,
)

# * model training
history = model.fit(
    train, epochs=200, validation_data=test, verbose=1, callbacks=[tb, red_lr, ckpt],
)

