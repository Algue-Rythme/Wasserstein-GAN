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
from keras.optimizers import RMSprop

# The Wassertein-Distance
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

# Simple model to that generate output of dimension data_dim with a noise of dimension noise_dim
def mlp_generator(noise_dim, data_dim, name='mlp_generator'):
    model = Sequential(name=name)
    model.add(Dense(128, input_dim=noise_dim, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(data_dim, activation='linear'))
    return model

# The critic is the discriminator, that takes a Tensor of dimension data_dim
# output either +1 (res. -1) if the example is fake (res. real)
def mlp_critic(data_dim, name='mpl_critic'):
    model = Sequential(name=name)
    model.add(Dense(128, input_dim=data_dim, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(1, activation='linear'))
    return model

# Combine the noise, the generator and the discriminator in order to build the GAN
def get_GAN(generator, critic, noise_dim, data_dim):
    gen_input = Input(shape=noise_dim, name="noise_input")
    generated = generator(gen_input)
    GAN_output = critic(generated)
    model = Model(inputs=[gen_input], outputs=[GAN_output], name="GAN")
    return model

# Clipping the weight is the heart of this algorithm
def clip_weights(critic, min_value, max_value):
    for layer in critic.layers:
        weights = layer.get_weights()
        K.clip(weights, min_alue, max_value)
        layer.set_weights(weights)

def train_wgan(generator, critic, noise_dim, nbEpochs, nbBatchPerEpochs, batchSize, eta_critic):
    generator.compile(loss='mse', optimizer=RMSprop)
    critic.compile(loss=wassertein_distance, optimizer=RMSprop)
    GAN = get_GAN(generator, critic, noise_dim, data_dim) #TODO
    for i_epoch in range(nbEpochs):
        for i_batch in range(nbBatchPerEpochs):
            list_critic_real_loss, list_critic_gen_loss = [], []
            for i_critic in range(eta_critic):
                clip_weights(critic)
                real_batch = batch_real_distribution(batchSize, i_batch) #TODO
                gen_batch = batch_generated_distribution(generator, batchSize) #TODO
                critic_real_loss = critic.train_on_batch(real_batch, -np.ones(real_batch.shape[0]))
                critic_gen_loss = critic.train_on_batch(gen_batch, np.ones(gen_batch.shape[0]))
                list_critic_real_loss.append(critic_real_loss)
                list_critic_gen_loss.append(critic_gen_loss)
            noise = get_noise(noise_dim, batchSize) #TODO
            # When we train the GAN, we want to train the weights that belong to
            # the generator, not to the critic
            critic.trainable = False
            gen_loss = GAN.train_on_batch(noise, -np.ones(noise.shape[0]))
            critic.trainable = True
