#%%
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

import matplotlib.pylab as plt
import random
import numpy as np

#%%
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
print(x_train.shape)

#%%
random_idx = random.sample(range(x_train.shape[0]),k=1)
print(random_idx[0])
plt.imshow(x_train[random_idx[0]],cmap='gray')
plt.show()

#%%
x_train, x_test = x_train[..., tf.newaxis], x_test[..., tf.newaxis]
print(x_train.shape)

#%%
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


#%%
print(x_train.shape)
print(y_train.shape)

#%%
