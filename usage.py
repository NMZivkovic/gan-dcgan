import numpy as np
from keras.datasets import fashion_mnist

from image_helper import ImageHelper
from gan import GAN


(X, _), (_, _) = fashion_mnist.load_data()
X_train = X / 127.5 - 1.
X_train = np.expand_dims(X_train, axis=3)

image_helper = ImageHelper()
generative_advarsial_network = GAN(X_train[0].shape, 100, image_helper)
generative_advarsial_network.train(30000, X_train, batch_size=32)