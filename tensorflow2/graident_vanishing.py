import tensorflow as tf 
import numpy as np 

def get_batch(x, y, batch_size):
    idxes = np.random.randint(0, len(y), batch_size)
    return x[idxes,:,:], y[idxes]

class model(object):
    
    def __init__(self, num_layer, num_filter, activation):
        self.num_layer = num_layer
        self.num_filter = num_filter
        self.nn_model = tf.keras.Sequential()

        for i in range(num_layer):
            self.nn_model.add(tf.keras.layers.Dense(num_filter, activation=activation, name=f"DenseLayer{i+1}"))

        self.nn_model.add(tf.keras.layers.Dense(10, name='OutputLayer'))


    @tf.function()
    def feed_forward(self, input_x):
        x = tf.cast(input_x, dtype=tf.float32)
        x = tf.reshape(x, (-1, 28*28))
        logits = self.nn_model(x)
        return logits

    
    def gradient_log(self, gradients, train_writer, step):
        assert len(gradients) == len(self.nn_model.trainable_variables)
        for i in range(len(gradients)):
            if 'kernel' in self.nn_model.trainable_variables[i].name:
                with train_writer.as_default():
                    tf.summary.scalar(f'mean_{int(i/2+1)}', tf.reduce_mean(tf.abs(gradients[i])), step=step)
                    tf.summary.histogram(f'histogram_{int(i/2+1)}', gradients[i], step=step)
                    tf.summary.histogram(f'histogram_weight_{int(i/2+1)}', self.nn_model.trainable_variables[i], step=step)


    @staticmethod
    def loss(logits, input_y):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=input_y))



if __name__ == '__main__':
    # load mnist dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0

    # initial parameters
    epochs = 10
    batch_size = 500
    total_batch = int(len(y_train)/batch_size)


