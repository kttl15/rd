import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint


# * set gpu memory usage
device = tf.config.list_physical_devices("GPU")[0]
tf.config.experimental.set_memory_growth(device, True)

tf.get_logger().setLevel("ERROR")


class Model:
    def __init__(self, img_shape):
        self.base_model_name = None
        self.IMG_SHAPE = img_shape

        self.data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
                tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
            ]
        )

    def __is_model_set__(self):
        """Check if model has been specified.

        Raises:
            NameError: base model has not been set prior to building the model.
        """
        if not self.base_model_name:
            raise NameError("base model not set. Please load a base model first")

    def load_model(self, base_model_name: str):
        """Load/download a model.

        Args:
            base_model_name (str): ['efficientnet', 'resnet50', 'vgg16']

        Raises:
            ValueError: error if base model name not in list
        """
        if base_model_name not in ["efficientnet", "resnet50", "vgg16"]:
            raise ValueError(
                f"{base_model_name} not one of ['efficientnet', 'resnet50', 'vgg16']"
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

    def build_model(self, train_data):
        """Returns a tf.keras.Model model.

        Returns:
            tf.keras.Model
        """
        self.__is_model_set__()
        # img_batch, _ = next(iter(train_data))
        # self.feature_batch = self.base_model(img_batch)
        inputs = tf.keras.layers.Input(shape=self.IMG_SHAPE)
        x = self.data_augmentation(inputs)
        x = self.preprocessor(x)
        x = self.base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        outputs = tf.keras.layers.Dense(189)(x)
        model = tf.keras.Model(inputs, outputs)
        return model


# * read data
PATH = "data/"
train_dir = os.path.join(PATH, "train")
validation_dir = os.path.join(PATH, "test")

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

train = image_dataset_from_directory(
    train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
)

val = image_dataset_from_directory(
    validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
)

# * enables prefetching. Prepares next training batch while model is training on current batch.
AUTOTUNE = tf.data.experimental.AUTOTUNE
train = train.prefetch(buffer_size=AUTOTUNE)
val = val.prefetch(buffer_size=AUTOTUNE)
test = test.prefetch(buffer_size=AUTOTUNE)

# * init model
model_name = "efficientnet"
model = Model(img_shape=IMG_SIZE + (3,))
model.load_model(model_name)
model = model.build_model(train)
base_learning_rate = 1e-3
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.summary()

# * callbacks
tb = TensorBoard(log_dir="logs", update_freq="epoch", write_graph=True)

red_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=5, verbose=1, min_lr=1e-5
)

if "ckpt" not in os.listdir():
    os.mkdir("ckpt")

curr_time = time.time()
ckpt = ModelCheckpoint(
    filepath=f"ckpt/weights_{model_name}_{curr_time}.h5",
    save_weights_only=False,
    monitor="val_accuracy",
    save_best_only=True,
)

# * model training
history = model.fit(
    train, epochs=200, validation_data=val, verbose=1, callbacks=[tb, red_lr, ckpt],
)

