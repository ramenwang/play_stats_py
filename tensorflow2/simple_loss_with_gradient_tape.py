import tensorflow as tf
import random  

def train_step(x):
    with tf.GradientTape() as tape:
        # needs to add watch method otherwise gradient will not be traced
        tape.watch(x)
        loss = tf.math.abs(a*x - b)
    dx = tape.gradient(loss, x)
    print(f"x = {x.numpy():.3f}, loss = {dx.numpy():.3f}")
    # print(dx)
    x.assign(x - dx.numpy())

if __name__ == '__main__':
    a, b = tf.constant(3.0), tf.constant(6.0)
    x = tf.Variable(random.randint(10,20)*1.0)
    # tf.cast cannot be used otherwise, the assign function would then be disabled

    print(f'The initial x is {x.numpy():.3f}')
    for i in range(7):
        train_step(x)
    
    print(x)