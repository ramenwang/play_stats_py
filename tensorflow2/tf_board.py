import tensorflow as tf 
import numpy as np 
import datetime as dt

@tf.function()
def nn_model(x_input, labels, w1, b1, w2, b2):
    
    x_input = tf.reshape(x_input, (x_input.shape[0], -1))

    with tf.name_scope('Hidden') as scope:
        x = tf.cast(x_input, dtype=tf.float32)
        hidden_logtis = tf.add(tf.matmul(x, w1), b1)
        hidden_out = tf.nn.sigmoid(hidden_logtis)

    with tf.name_scope('Output') as scope:
        logits = tf.add(tf.matmul(hidden_out, w2), b2)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    return logits, hidden_logtis, hidden_out, cross_entropy


def get_batch(input_x, input_y, batch_size):
    '''
    input_x has a shape of [..., 28, 28]
    input_y is a one dimensional tuple
    '''
    idxs = np.random.randint(low=0, high=len(input_y), size=batch_size)
    return input_x[idxs,:,:], input_y[idxs]


if __name__ == "__main__":
    # load data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize data
    x_train, x_test = x_train/255.0, x_test/255.0

    epochs = 10
    batch_size = 1000
    optimizer = tf.optimizers.Adam()
    total_batch = int(len(y_train)/batch_size)

    # initialize the weights that connecting the input to the hidden layers
    w1 = tf.Variable(tf.random.normal([784, 300], stddev=0.03), name='w1')
    b1 = tf.Variable(tf.random.normal([300]), name='b1')
    w2 = tf.Variable(tf.random.normal([300, 10], stddev=0.03), name='w2')
    b2 = tf.Variable(tf.random.normal([10]), name='b2')

    out_file = "tensorflow2/tensorboard" + \
        f"/TensorFlow_Visualization_{dt.datetime.now().strftime('%d%m%Y%H%M')}"
    train_summary_writer = tf.summary.create_file_writer(out_file)

    for epoch in range(epochs):
        ave_loss = 0
        for _ in range(total_batch):
            x_batch, y_batch = get_batch(x_train, y_train, batch_size)
            x_batch, y_batch = tf.Variable(x_batch), tf.one_hot(y_batch,10)

            with tf.GradientTape() as tape:
                logtis, hidden_logits, hidden_out, loss = nn_model(x_batch, y_batch, w1, b1, w2, b2)
            
            gradients = tape.gradient(loss, [w1, b1, w2, b2])
            optimizer.apply_gradients(zip(gradients, [w1, b1, w2, b2]))
            ave_loss += loss/total_batch
        
        test_logits,_,_,_ = nn_model(x_test, tf.one_hot(y_test, 10), w1, b1, w2, b2)
        max_idx = tf.argmax(test_logits, axis=1)
        test_acc = np.sum(max_idx.numpy()==y_test) / len(y_test)
        print(f"Epoch: {epoch + 1}, loss={ave_loss:.3f}, test set accuracy={test_acc*100:.3f}%")
        
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', ave_loss, step=epoch)
            tf.summary.scalar('accuracy', test_acc, step=epoch)
            tf.summary.histogram("Hidden_logits", hidden_logits, step=epoch)
            tf.summary.histogram("Hidden_output", hidden_out, step=epoch)

    tf.summary.trace_on(graph = True)
    logits, _, _, _ = nn_model(x_batch, y_batch, w1, b1, w2, b2)
    with train_summary_writer.as_default():
        tf.summary.trace_export(name='graph', step=0)
