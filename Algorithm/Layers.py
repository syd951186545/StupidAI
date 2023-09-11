from keras import Sequential
from keras.layers import TimeDistributed, Conv2D, Dropout, Flatten, Dense, Attention, Multiply, Add
import tensorflow as tf


def ActorGlobalLiner():
    block = Sequential([
        Dense(256, activation='relu'),
        Dropout(0.2),
    ], name="ActorGlobalLiner")
    return block


def ActorLoacalLiner():
    block = TimeDistributed(Sequential([
        Dense(256, activation='relu'),
        Dropout(0.2),
    ], name="ActorLoacalLiner"))
    return block


def OutputLiner():
    block = TimeDistributed(Sequential([
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(9, activation='softmax')
    ], name="OutputLiner"))
    return block


def CriticLiner():
    block = TimeDistributed(Sequential([
        Dense(256, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ], name="output"))
    return block


def CNN_strides2(name, input_channels):
    _block = TimeDistributed(Sequential([
        Conv2D(32, (3, 3), input_shape=(21, 21, input_channels), activation='relu'),
        Conv2D(64, (3, 3), strides=2, activation='relu'),
        # AveragePooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), strides=2, activation='relu'),
        # AveragePooling2D(pool_size=(2, 2)),
        Dropout(0.2),
    ]), name=name)
    return _block


def CNN(name, input_channels):
    _block = Sequential([
        Conv2D(32, (3, 3), input_shape=(21, 21, input_channels), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Dropout(0.1),
        # AveragePooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        # AveragePooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Flatten()
    ], name=name)
    return _block


def TimeDistributed_CNN(name, input_channels):
    _block = TimeDistributed(CNN(name, input_channels))
    return _block


# 注意力层
class AttentionModule(tf.keras.layers.Layer):
    def __init__(self, units=64, **kwargs):
        super(AttentionModule, self).__init__(**kwargs)
        self.units = units
        self.q_conv = tf.keras.layers.Conv2D(self.units, (1, 1), activation='relu')
        self.k_conv = tf.keras.layers.Conv2D(self.units, (1, 1), activation='relu')
        self.v_conv = tf.keras.layers.Conv2D(self.units, (1, 1), activation='relu')
        self.flatten = Flatten()

    def call(self, inputs):
        q = self.q_conv(inputs)
        k = self.k_conv(inputs)
        v = self.v_conv(inputs)

        attn_score = tf.keras.layers.Attention()([q, k])
        attn_output = tf.keras.layers.Multiply()([v, attn_score])
        attn_res = tf.keras.layers.Add()([inputs, attn_output])
        return self.flatten(attn_res)

    def get_config(self):
        config = super(AttentionModule, self).get_config()
        config.update({'units': self.units})
        return config
