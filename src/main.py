#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import time

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input
from keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Reshape
from keras.layers.convolutional import Conv2D, Deconv2D, UpSampling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import RMSprop
from keras.utils import Progbar

def wassertein_distance(y_true, y_pred):
    return K.mean(y_true * y_pred)

# Utility
def visualize_model(model):
    model.summary()
    from keras.utils import plot_model
    plot_model(model,
               to_file='figures/%s.png' % model.name,
               show_shapes=True,
               show_layer_names=True)

def mlp_generator(noise_dim, data_dim, name='mlp_generator'):
    """
    Simple model that generates output of dimension
        data_dim with a noise of dimension noise_dim
    """

    model = Sequential(name=name)
    model.add(Dense(128, input_shape=noise_dim, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(data_dim, activation='linear'))
    return model

def mlp_critic(data_dim, name='mlp_critic'):
    """ 
    The critic is the discriminator, that takes a Tensor of dimension data_dim
    output either +1 (res. -1) if the example is fake (res. real)
    """
    model = Sequential(name=name)
    model.add(Dense(128, input_dim=data_dim, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(1, activation='linear'))
    return model

def get_GAN(generator, critic, noise_dim, data_dim):
    """
    Combine the noise, the generator and the discriminator in order to build the GAN
    """
    gen_input = Input(shape=noise_dim, name="noise_input")
    generated = generator(gen_input)
    GAN_output = critic(generated)
    model = Model(inputs=[gen_input], outputs=[GAN_output], name="GAN")
    return model

def clip_weights(critic, min_value, max_value):
    """ 
    Clipping the weight is the heart of this algorithm
    -> back to a case where our weights lay in a compact
    """
    for layer in critic.layers:
        weights = layer.get_weights()
        weights = [np.clip(w, min_value, max_value) for w in weights]
        layer.set_weights(weights)

def batch_real_distribution(batchSize, i_batch, data_dim):
    return np.random.normal(4, 2, (batchSize, data_dim))

def get_noise(noise_dim, batchSize):
    return np.random.normal(0, 1, (batchSize,) + noise_dim)

def batch_generated_distribution(generator, batchSize, noise_dim):
     noise = get_noise(noise_dim, batchSize)
     return generator.predict(noise)

def train_wgan(generator, critic, noise_dim, data_dim, nbEpochs, nbBatchPerEpochs, batchSize, eta_critic):
    epoch_size = nbBatchPerEpochs * batchSize
    GAN = get_GAN(generator, critic, noise_dim, data_dim)
    generator.compile(loss='mse', optimizer=RMSprop())
    critic.trainable = False
    GAN.compile(loss=wassertein_distance, optimizer=RMSprop())
    critic.trainable = True
    critic.compile(loss=wassertein_distance, optimizer=RMSprop())
    for i_epoch in range(nbEpochs):
        progbar = Progbar(epoch_size)
        start = time.time()
        for i_batch in range(nbBatchPerEpochs):
            list_critic_real_loss, list_critic_gen_loss = [], []
            for i_critic in range(eta_critic):
                clip_weights(critic, -0.01, 0.01)
                real_batch = batch_real_distribution(batchSize, i_batch, data_dim)
                gen_batch = batch_generated_distribution(generator, batchSize, noise_dim)
                critic_real_loss = critic.train_on_batch(real_batch, -np.ones(real_batch.shape[0]))
                critic_gen_loss = critic.train_on_batch(gen_batch, np.ones(gen_batch.shape[0]))
                list_critic_real_loss.append(critic_real_loss)
                list_critic_gen_loss.append(critic_gen_loss)
            noise = get_noise(noise_dim, batchSize)
            # When we train the GAN, we want to train the weights that belong to
            # the generator, not to the critic
            critic.trainable = False
            gen_loss = GAN.train_on_batch(noise, -np.ones(noise.shape[0]))
            critic.trainable = True
            progbar.add(batchSize, values=[("Loss_D", -np.mean(list_critic_real_loss) - np.mean(list_critic_gen_loss)),
                                            ("Loss_D_real", -np.mean(list_critic_real_loss)),
                                            ("Loss_D_gen", np.mean(list_critic_gen_loss)),
                                            ("Loss_G", -gen_loss)])
        print('\nEpoch %s/%s, Time: %s' % (i_epoch + 1, nbEpochs, time.time() - start))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--train', action='store_true', help="run training phase")
    parser.add_argument('--noise_dim', default=(10,), type=tuple, help="TODO")
    parser.add_argument('--data_dim', default=10, type=int, help="TODO")
    parser.add_argument('--nb_epoch', default=2, type=int, help="Number of epochs")
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--n_batch_per_epoch', default=300, type=int, help="Number of batch per epochs")
    parser.add_argument('--eta_critic', default=5, type=int, help="TODO")
    args = parser.parse_args()

    noise_dim = (10,)
    data_dim = 10
    param_filename = "param"
    generator = mlp_generator(noise_dim, data_dim)
    if args.train or not os.path.exists(param_filename):
        critic = mlp_critic(data_dim)
        train_wgan(generator, critic, args.noise_dim, args.data_dim, args.nb_epoch, args.n_batch_per_epoch, args.batch_size, args.eta_critic)
    else:
        generator.load_weights(param_filename)
    generator.summary()
    generator.predict()
    generator.save_weights(param_filename)
