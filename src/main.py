import numpy as np
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input
from keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Reshape
from keras.layers.convolutional import Conv2D, Deconv2D, UpSampling2D
from keras.layers.pooling import GlobalAveragePooling2D

def wassertein_distance(y_true, y_pred):
    return K.mean(y_true * y_pred)

def visualize_model(model):
    model.summary()
    from keras.utils import plot_model
    plot_model(model,
               to_file='figures/%s.png' % model.name,
               show_shapes=True,
               show_layer_names=True)

def mlp_generator(noise_dim, data_dim, name='mlp_generator'):
    model = Sequential(name=name)
    model.add(Dense(128, input_dim=noise_dim, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(data_dim, activation='linear'))
    return model

model = mlp_generator(5, 2)
visualize_model(model)

def mlp_discriminator(data_dim, name='mpl_discriminator'):
    model = Sequential(name=name)
    model.add(Dense(128, input_dim=data_dim, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(1, activation='linear'))
    return model

dis = mlp_discriminator(2)
visualize_model(dis)
