import tensorflow as tf
import numpy as np 

# three layer class
class nn_model():
    
    def __init__(self):
        # initialize the class with weights matrix
        self.w1 = tf.Variable(tf.random.normal([28*28, 300]), name='w1')
        self.b1 = tf.Variable(tf.random.normal([300]), name='b1')
        self.w2 = tf.Variable(tf.random.normal([300, 10]), name='w2')
        self.b2 = tf.Variable(tf.random.normal([10]), name='b2')

    def build_nn(self, input_x):
        '''
        input_x is a 3D tuple with the shape of [batch_size, 28, 28]
        '''
        x = tf.reshape(input_x, (input_x.shape[0], -1))
        x = tf.cast(x, dtype=tf.float32)
        x = tf.add(tf.matmul(x, self.w1), self.b1)
        x = tf.nn.relu(x)
        logits = tf.add(tf.matmul(x, self.w2), self.b2)
        return logits
        

def loss_fn(logits, labels):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return cross_entropy

def get_batch(input_x, input_y, batch_size):
    '''
    input_x has a shape of [..., 28, 28]
    input_y is a one dimensional tuple
    '''
    idxs = np.random.randint(low=0, high=len(input_y), size=batch_size)
    return input_x[idxs,:,:], input_y[idxs]

if __name__ == '__main__':
    # load data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize data
    x_train, x_test = x_train/255.0, x_test/255.0

    epochs = 10
    batch_size = 100
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    total_batch = int(len(y_train)/batch_size)
    nn = nn_model()  
    
    for epoch in range(epochs):
        ave_loss = 0
        for i in range(total_batch):
            batch_x, batch_y = get_batch(x_train, y_train, batch_size)
            batch_x, batch_y = tf.Variable(batch_x), tf.one_hot(tf.Variable(batch_y), 10)

            with tf.GradientTape() as tape:
                logits = nn.build_nn(batch_x)
                loss = loss_fn(logits, batch_y)

            gradients = tape.gradient(loss, [nn.w1, nn.b1, nn.w2, nn.b2])
            optimizer.apply_gradients(zip(gradients, [nn.w1, nn.b1, nn.w2, nn.b2]))
            ave_loss += loss / total_batch

        test_logits = nn.build_nn(x_test)
        max_idx = tf.argmax(test_logits, axis=1)
        acc = np.sum(max_idx.numpy() == y_test)/len(y_test)

        print(f"Epoch: {epoch}, Loss = {ave_loss:.3f}, Accuracy = {acc:.3f}")

    print('Done!')



