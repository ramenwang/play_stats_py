#%%
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
print(x_train[0])

#%%
x_train, x_test = x_train/255.0, x_test/255.0
# y_train, y_test = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)
print(y_train[0])

#%%
plt.imshow(x_train[0], cmap='gray')
plt.show()

#%%
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#%%
model.fit(x_train, y_train, epochs=4, use_multiprocessing = True)
model.evaluate(x_test, y_test, verbose=2)

#%%
