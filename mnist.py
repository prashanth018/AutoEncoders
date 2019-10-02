import os
from AutoEncoder import AutoEncoder
from VariationalAutoEncoder import VariationalAutoEncoder
from keras.datasets import mnist
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype('float32') / 255
    x_test = x_test.reshape(x_test.shape + (1,))

    return (x_train, y_train), (x_test, y_test)


def load_model(model_class, folder):
    with open(os.path.join(folder, 'params/params.pkl'), 'rb') as f:
        params = pickle.load(f)

    model = model_class(*params)
    model.load_weights(os.path.join(folder, 'weights/weights.h5'))

    return model


SECTION = 'vae'
RUN_ID = '0002'
DATA_NAME = 'digits'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))
    os.mkdir(os.path.join(RUN_FOLDER, 'params'))

MODE = 'build'

(x_train, y_train), (x_test, y_test) = load_mnist()

AE = AutoEncoder(input_dim=(28, 28, 1), enc_conv_filters=[32, 64, 64, 64], enc_conv_kernal=[3, 3, 3, 3],
                  enc_conv_strides=[1, 2, 2, 1], dec_deconv_filters=[64, 64, 32, 1],
                  dec_deconv_kernal=[3, 3, 3, 3], dec_deconv_strides=[1, 2, 2, 1], z_dim=2)

VAE = VariationalAutoEncoder(input_dim=(28, 28, 1), enc_conv_filters=[32, 64, 64, 64], enc_conv_kernal=[3, 3, 3, 3],
                  enc_conv_strides=[1, 2, 2, 1], dec_deconv_filters=[64, 64, 32, 1],
                  dec_deconv_kernal=[3, 3, 3, 3], dec_deconv_strides=[1, 2, 2, 1], z_dim=2)

if MODE == 'build':
    VAE.save(RUN_FOLDER)
else:
    VAE.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

VAE.encoder.summary()
VAE.decoder.summary()
#
LEARNING_RATE = 0.0005
R_LOSS_FACTOR = 1000
BATCH_SIZE = 32
EPOCHS = 100
PRINT_EVERY_N_BATCHES = 100
INITIAL_EPOCH = 0

VAE.compile(LEARNING_RATE, R_LOSS_FACTOR)
#
VAE.train(x_train[:30000], batch_size=BATCH_SIZE, epochs=EPOCHS, run_folder=RUN_FOLDER,
          print_every_n_batches=PRINT_EVERY_N_BATCHES, initial_epoch=INITIAL_EPOCH)


# AE = load_model(AutoEncoder, RUN_FOLDER)


def Reconstruction():
    n_to_show = 10

    example_idx = np.random.choice(range(len(x_test)), n_to_show)
    example_images = x_test[example_idx]

    z_points = AE.encoder.predict(example_images)
    reconst_images = AE.decoder.predict(z_points)

    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(n_to_show):
        img = example_images[i].squeeze()
        ax = fig.add_subplot(2, n_to_show, i + 1)
        ax.axis('off')
        ax.text(0.5, -0.35, str(np.round(z_points[i], 1)), fontsize=10, ha='center', transform=ax.transAxes)
        ax.imshow(img, cmap='gray_r')

    for i in range(n_to_show):
        img = reconst_images[i].squeeze()
        ax = fig.add_subplot(2, n_to_show, i + n_to_show + 1)
        ax.axis('off')
        ax.imshow(img, cmap='gray_r')

    plt.savefig('Reconstruction.png')


def EncoderPrediction():
    n_to_show = 5000
    grid_size = 15
    figsize = 12

    example_idx = np.random.choice(range(len(x_test)), n_to_show)
    example_images = x_test[example_idx]
    example_labels = y_test[example_idx]

    z_points = AE.encoder.predict(example_images)

    min_x = min(z_points[:, 0])
    max_x = max(z_points[:, 0])
    min_y = min(z_points[:, 1])
    max_y = max(z_points[:, 1])

    plt.figure(figsize=(figsize, figsize))
    plt.scatter(z_points[:, 0], z_points[:, 1], cmap='rainbow', c=example_labels
                , alpha=0.5, s=2)
    plt.colorbar()
    plt.savefig('EncoderPrediction.png')


def ImageGeneration():
    grid_size = 10
    grid_depth = 3
    figsize = 15
    n_to_show = 5000

    example_idx = np.random.choice(range(len(x_test)), n_to_show)
    example_images = x_test[example_idx]
    z_points = AE.encoder.predict(example_images)

    min_x = min(z_points[:, 0])
    max_x = max(z_points[:, 0])
    min_y = min(z_points[:, 1])
    max_y = max(z_points[:, 1])

    x = np.random.uniform(min_x, max_x, size=grid_size * grid_depth)
    y = np.random.uniform(min_y, max_y, size=grid_size * grid_depth)
    z_grid = np.array(list(zip(x, y)))

    reconst = AE.decoder.predict(z_grid)
    plt.scatter(z_grid[:, 0], z_grid[:, 1], c='red', alpha=1, s=20)
    plt.savefig('ImageGeneration_EncoderPrediction.png')

    fig = plt.figure(figsize=(figsize, grid_depth))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(grid_size * grid_depth):
        ax = fig.add_subplot(grid_depth, grid_size, i + 1)
        ax.axis('off')
        ax.text(0.5, -0.35, str(np.round(z_grid[i], 1)), fontsize=10, ha='center', transform=ax.transAxes)

        ax.imshow(reconst[i, :, :, 0], cmap='Greys')

    plt.savefig('ImageGeneration_DecoderPrediction.png')


def ImageReconstruction():
    n_to_show = 5000
    grid_size = 20
    figsize = 8

    example_idx = np.random.choice(range(len(x_test)), n_to_show)
    example_images = x_test[example_idx]
    example_labels = y_test[example_idx]

    z_points = AE.encoder.predict(example_images)

    x = np.linspace(min(z_points[:, 0]), max(z_points[:, 0]), grid_size)
    y = np.linspace(max(z_points[:, 1]), min(z_points[:, 1]), grid_size)

    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    z_grid = np.array(list(zip(xv, yv)))

    reconst = AE.decoder.predict(z_grid)

    plt.scatter(z_grid[:, 0], z_grid[:, 1], c='black'  # , cmap='rainbow' , c= example_labels
                , alpha=1, s=5)

    plt.savefig('temp.png')

    fig = plt.figure(figsize=(figsize, figsize))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(grid_size ** 2):
        ax = fig.add_subplot(grid_size, grid_size, i + 1)
        ax.axis('off')
        ax.imshow(reconst[i, :, :, 0], cmap='Greys')

    plt.savefig('temp2.png')


if __name__ == '__main__':
    # Reconstruction()
    # EncoderPrediction()
    # ImageGeneration()
    # ImageReconstruction()
    # AE.plot_model(RUN_FOLDER)
    pass