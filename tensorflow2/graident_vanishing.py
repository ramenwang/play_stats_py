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
            self.nn_model.add(tf.keras.layers.Dense(num_filter, activation=activation, name=f"DenseLayer_{i+1}"))

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

    def plot_computation_graph(self, train_writer, x_batch):
        tf.summary.trace_on(graph = True)
        self.feed_forward(input_x = x_batch)
        with train_writer.as_default():
            tf.summary.trace_export(name='graph', step=0)


def train(Model:model, sub_folder:str, iterations:int=1200, batch_size:int=32, log_freq:int=200):
    train_writer = tf.summary.create_file_writer('./tensorflow2/tensorboard'+'/'+sub_folder)
    Model.plot_computation_graph(train_writer, x_batch=x_train[batch_size,:,:])
    optimizor = tf.optimizers.Adam()
    for i in range(iterations):
        x_batch, y_batch = get_batch(x_train, y_train, batch_size)
        x_batch, y_batch = tf.Variable(x_batch), tf.cast(y_batch, dtype=tf.int32)
        with tf.GradientTape() as tape:
            logits = Model.feed_forward(x_batch)
            loss = Model.loss(logits, y_batch)
        gradients = tape.gradient(loss, Model.nn_model.trainable_variables)
        optimizor.apply_gradients(zip(gradients, Model.nn_model.trainable_variables))
        if i % log_freq == 0:
            max_idxs = tf.argmax(logits, axis=1)
            acc = np.sum(max_idxs.numpy()==y_batch.numpy()) / len(y_batch.numpy())
            print(f"Interation = {i} ; Loss = {loss:.3f} ; Accuracy = {100 * acc:.3f}%")
            with train_writer.as_default():
                tf.summary.scalar('loss', loss, step=i)
                tf.summary.scalar('accuracy', acc, step=i)
            Model.gradient_log(gradients, train_writer, step=i)




if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    
    # load mnist dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0

    # initial parameters
    scenarios = ['sigmoid','relu','leaky_relu']
    activations = [tf.nn.sigmoid, tf.nn.relu, tf.nn.leaky_relu]

    for i in range(len(scenarios)):
        print('Running on scenario = ' + scenarios[i])
        iModel = model(num_layer=6, num_filter=10, activation=activations[i])
        train(Model=iModel, sub_folder=scenarios[i])


