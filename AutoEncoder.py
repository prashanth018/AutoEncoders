from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Dropout, Flatten, Dense, Reshape, \
    Conv2DTranspose, Activation
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
import matplotlib.pyplot as plt
from utils.callback import CustomCallback, step_decay_schedule

import numpy as np
import json
import os
import pickle


class AutoEncoder():
    def __init__(self, input_dim, enc_conv_filters, enc_conv_kernal, enc_conv_strides, dec_deconv_filters,
                 dec_deconv_kernal, dec_deconv_strides, use_batch_norm=False, use_dropout=False, z_dim=2):
        self.name = 'autoencoder'
        self.input_dim = input_dim
        self.enc_conv_filters = enc_conv_filters
        self.enc_conv_kernal = enc_conv_kernal
        self.enc_conv_strides = enc_conv_strides

        self.dec_deconv_filters = dec_deconv_filters
        self.dec_deconv_kernal = dec_deconv_kernal
        self.dec_deconv_strides = dec_deconv_strides

        self.z_dim = z_dim

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.num_layers_encoder = len(self.enc_conv_filters)
        self.num_layers_decoder = len(self.dec_deconv_filters)
        self._build()

    def _build(self):
        encoder_input = Input(shape=self.input_dim, name='encoder_input')
        x = encoder_input

        for _ in range(self.num_layers_encoder):
            conv_layer = Conv2D(filters=self.enc_conv_filters[_], kernel_size=self.enc_conv_kernal[_],
                                strides=self.enc_conv_strides[_], padding='same', name='encoder_conv_' + str(_))
            x = conv_layer(x)
            x = LeakyReLU()(x)

            if self.use_batch_norm:
                x = BatchNormalization()(x)

            if self.use_dropout:
                x = Dropout(rate=0.25)(x)

        shape_before_flattening = K.int_shape(x)[1:]

        x = Flatten()(x)

        encoder_output = Dense(self.z_dim, name='encoder_output')(x)

        self.encoder = Model(encoder_input, encoder_output)

        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')

        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        for _ in range(self.num_layers_decoder):
            conv_t_layer = Conv2DTranspose(filters=self.dec_deconv_filters[_], kernel_size=self.dec_deconv_kernal[_],
                                           strides=self.dec_deconv_strides[_], padding='same',
                                           name='decoder_conv_t' + str(_))
            x = conv_t_layer(x)

            if _ < self.num_layers_decoder - 1:
                x = LeakyReLU()(x)

                if self.use_batch_norm:
                    x = BatchNormalization()(x)

                if self.use_dropout:
                    x = Dropout(0.25)(x)
            else:
                x = Activation('sigmoid')(x)

        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output)

        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)

    def compile(self, learning_rate):
        self.learning_rate = learning_rate
        optimizer = Adam(lr=learning_rate)

        def r_loss(y_true, y_pred):
            return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])

        self.model.compile(optimizer=optimizer, loss=r_loss)

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'weights'))
            os.makedirs(os.path.join(folder, 'images'))

        with open(os.path.join(folder, 'params/params.pkl'), 'wb') as f:
            pickle.dump([self.input_dim, self.enc_conv_filters, self.enc_conv_kernal, self.enc_conv_strides,
                         self.dec_deconv_filters, self.dec_deconv_kernal, self.dec_deconv_strides,
                         self.use_batch_norm, self.use_dropout, self.z_dim], f)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches=100, initial_epoch=0, lr_decay=1):
        custom_callback = CustomCallback(run_folder, print_every_n_batches, initial_epoch, self)
        lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint2 = ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'), save_weights_only=True, verbose=1)

        callbacks_list = [checkpoint2, custom_callback, lr_sched]

        self.model.fit(x_train, x_train, batch_size=batch_size, shuffle=True, epochs=epochs,
                       initial_epoch=initial_epoch, callbacks=callbacks_list)

    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder, 'viz/model.png'), show_shapes=True,
                   show_layer_names=True)
        plot_model(self.encoder, to_file=os.path.join(run_folder, 'viz/encoder.png'), show_shapes=True,
                   show_layer_names=True)
        plot_model(self.decoder, to_file=os.path.join(run_folder, 'viz/decoder.png'), show_shapes=True,
                   show_layer_names=True)
