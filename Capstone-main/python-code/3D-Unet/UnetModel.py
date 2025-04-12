from typing import Concatenate
import tensorflow as tf
from keras import Input, Model
from keras.src.layers import Conv3D, BatchNormalization, Activation, MaxPooling3D, UpSampling3D, Concatenate
from UnetData import UnetData
from UnetData import TEST_DATA_PATH, TRAINING_DATA_PATH
from tqdm.keras import TqdmCallback

class UnetModel:
    def __init__(self, input_shape=(48, 48, 48, 1), num_classes=1, lr=0.0001, epochs=20, loss="binary_crossentropy"):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.loss = loss

        # Build and compile the model upon initialization
        self.model = self.unet_3d()
        self.compile_model()

    def conv_block(self, tf_inputs, num_filters):
        """Convolutional block with Conv3D + BatchNorm + ReLU"""
        x = Conv3D(num_filters, (3, 3, 3), padding="same")(tf_inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv3D(num_filters, (3, 3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x

    def unet_3d(self):
        """Builds a 3D U-Net model"""
        tf_inputs = Input(self.input_shape)

        # --- ENCODER ---
        c1 = self.conv_block(tf_inputs, 32)
        p1 = MaxPooling3D((2, 2, 2))(c1)

        c2 = self.conv_block(p1, 64)
        p2 = MaxPooling3D((2, 2, 2))(c2)

        c3 = self.conv_block(p2, 128)
        p3 = MaxPooling3D((2, 2, 2))(c3)

        c4 = self.conv_block(p3, 256)
        p4 = MaxPooling3D((2, 2, 2))(c4)

        # --- BRIDGE ---
        c5 = self.conv_block(p4, 512)

        # --- DECODER ---
        u6 = UpSampling3D((2, 2, 2))(c5)
        u6 = Concatenate()([u6, c4])
        c6 = self.conv_block(u6, 256)

        u7 = UpSampling3D((2, 2, 2))(c6)
        u7 = Concatenate()([u7, c3])
        c7 = self.conv_block(u7, 128)

        u8 = UpSampling3D((2, 2, 2))(c7)
        u8 = Concatenate()([u8, c2])
        c8 = self.conv_block(u8, 64)

        u9 = UpSampling3D((2, 2, 2))(c8)
        u9 = Concatenate()([u9, c1])
        c9 = self.conv_block(u9, 32)

        # --- OUTPUT LAYER ---
        outputs = Conv3D(self.num_classes, (1, 1, 1), activation="sigmoid")(c9)  # Sigmoid for binary segmentation

        return Model(tf_inputs, outputs)

    def compile_model(self):
        """Compiles the U-Net model"""
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                           loss=self.loss,
                           metrics=["accuracy"])

    def train(self, X_train, Y_train, batch_size=2):
        """Train the model using .fit()"""
        self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=self.epochs)

    def summary(self):
        """Prints the model summary"""
        self.model.summary()

unetdata = UnetData(TRAINING_DATA_PATH, TEST_DATA_PATH)

unetdata.sort_training_data()
unetdata.sort_testing_data()
unetdata.load_numpy_and_expand_x(unetdata.X_TRAIN_PATHS)
unetdata.load_numpy_and_expand_x(unetdata.X_TEST_PATHS)
unetdata.load_numpy_and_expand_y(unetdata.Y_TRAIN_PATHS)

X_train, Y_train = unetdata.patch_volumes_training(unetdata.X_TRAIN, unetdata.Y_TRAIN)
X_test = unetdata.patch_volumes_testing(unetdata.X_TEST)

unet_model = UnetModel()

unet_model.model.fit(X_train, Y_train, batch_size=1, epochs=20, callbacks=[TqdmCallback()])