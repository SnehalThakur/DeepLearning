import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Block(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, pool_size, dropout_rate):
        super(Block, self).__init__()
        self.C1 = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
        self.B1 = BatchNormalization()
        self.A1 = Activation('relu')
        self.P1 = MaxPooling2D(pool_size=pool_size, strides=2, padding=padding)
        self.Dr1 = Dropout(dropout_rate)

    def call(self, x):
        x = self.C1(x)
        x = self.B1(x)
        x = self.A1(x)
        x = self.P1(x)
        y = self.Dr1(x)
        return y