import numpy as np
from keras.datasets import fashion_mnist

from image_helper import ImageHelper
from dcgan import DCGAN


(X, _), (_, _) = fashion_mnist.load_data()
X_train = X / 127.5 - 1.
X_train = np.expand_dims(X_train, axis=3)

image_helper = ImageHelper()
generative_advarsial_network = DCGAN(X_train[0].shape, 100, image_helper, 1)
generative_advarsial_network.train(20000, X_train, batch_size=32)