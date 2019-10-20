# test the concept of autograph in tensorflow 2

import tensorflow as tf 

@tf.function
def identity(x):
    print('Graph created!')


x1 = tf.random.uniform((10,10))
x2 = tf.random.uniform((10,10))
print('calling identity function 1st')
identity(x=x1)
print('calling identity function 2nd')
identity(x=x2)

x3 = tf.constant(int(input()))
print('calling identity function 3rd')
identity(x3)