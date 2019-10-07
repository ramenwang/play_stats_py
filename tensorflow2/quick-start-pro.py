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
    (x_train, y_train)).shuffle(10000).batch(64)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)


#%%
print(train_ds)
print(train_ds)

#%%
class testModel(Model):
    def __init__(self):
        super(testModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation = 'relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation = 'relu')
        self.d2 = Dense(10, activation = 'softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x

model = testModel()

#%%
loss_object_function = tf.keras.losses.sparse_categorical_crossentropy()
optimizor = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


#%%
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object_function(labels, predictions)
    grediants = tape.grediants(loss, model.trainable_variables)

    train_loss(loss)
    train_accuracy(labels, predictions)



