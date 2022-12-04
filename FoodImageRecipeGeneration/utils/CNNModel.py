import os

import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils.Block import Block

epochs = 30
batch_size = 32
img_height, img_width = 300, 300
input_shape = (img_height, img_width, 3)


def create_data_sparse():
    data_gen = ImageDataGenerator(rescale=1.0 / 255)
    train_ds = data_gen.flow_from_directory('/kaggle/input/food11/food11/train/', target_size=(img_height, img_width),
                                            class_mode='sparse', batch_size=batch_size, subset='training')
    val_ds = data_gen.flow_from_directory('/kaggle/input/food11/food11/test/', target_size=(img_height, img_width),
                                          class_mode='sparse', batch_size=batch_size, shuffle=False)

    return train_ds, val_ds
# train_ds, val_ds = create_data_sparse()

class Network(tf.keras.Model):
    def __init__(self):
        super(Network, self).__init__()
        self.C1 = Conv2D(filters=32, kernel_size=(3 * 3), strides=1, padding='same', input_shape=input_shape)
        self.B1 = BatchNormalization()
        self.A1 = Activation('relu')

        self.layer1 = Block(filters=32, kernel_size=(3 * 3), strides=1, padding='same', pool_size=(2 * 2),
                            dropout_rate=0.2)
        self.layer2 = Block(filters=64, kernel_size=(3 * 3), strides=1, padding='same', pool_size=(2 * 2),
                            dropout_rate=0.4)
        self.layer3 = Block(filters=32, kernel_size=(3 * 3), strides=1, padding='same', pool_size=(2 * 2),
                            dropout_rate=0.3)

        self.F1 = Flatten()
        self.D1 = Dense(128, activation='relu')
        self.B2 = BatchNormalization()
        self.D2 = Dense(128, activation='relu')
        self.D3 = Dense(128, activation='relu')
        self.D4 = Dense(11, activation='softmax')

    def call(self, x):
        x = self.C1(x)
        x = self.B1(x)
        x = self.A1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.F1(x)
        x = self.D1(x)
        x = self.B2(x)
        x = self.D2(x)
        x = self.D3(x)
        y = self.D4(x)
        return y

    def __repr__(self):
        name = 'JiaoWoGuanRen_Model'
        return name


def create_model(shape):
    model = Network()
    model.call(shape)
    model.built = True
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['sparse_categorical_accuracy'])
    print(model.summary())
    return model

# checkpoint_save_path = './Model.ckpt'
# if os.path.exists(checkpoint_save_path + '.index'):
#     net.load_weights(checkpoint_save_path)
#
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path, save_weights_only=True,
#                                                  save_best_only=True)

# history = net.fit(train_ds, epochs=epochs, batch_size=batch_size, callbacks=[cp_callback])
# net.summary()