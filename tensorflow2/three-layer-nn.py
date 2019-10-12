#%%
import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0


#%%
def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)
    return x_data[idxs,:,:], y_data[idxs]

#%%
idxs = np.random.randint(0,10,5)
print(idxs)
print(np.arange(10)[idxs])

#%%
epochs = 10
batch_size = 100

# convert x test into tensor, because it will not go thru the batch process
print(type(x_test))
x_test = tf.Variable(x_test)
print(type(x_test))

#%%
# declear the weights that connecting the input to the hidden layers
w1 = tf.Variable(tf.random.normal([784, 300], stddev=0.03), name='w1')
b1 = tf.Variable(tf.random.normal([300]), name='b1')

w2 = tf.Variable(tf.random.normal([300, 10], stddev=0.03), name='w2')
b2 = tf.Variable(tf.random.normal([10]), name='b2')

#%%
# define function for feedforward network
def nn_model(x_input, w1, b1, w2, b2):
    # flatten input from 28 * 28 to 784
    x_input = tf.reshape(x_input, (x_input.shape[0], -1))
    x = tf.add(tf.matmul(tf.cast(x_input, tf.float32), w1), b1)
    x = tf.nn.relu(x)
    logits = tf.add(tf.matmul(x, w2), b2)
    return logits

def loss_fn(logits, labels):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return cross_entropy

optimizer = tf.keras.optimizers.Adam()

#%%

total_batch = int(len(y_train)/batch_size)

for epoch in range(epochs):
    avg_loss = 0
    for i in range(total_batch):
        batch_x, batch_y = get_batch(x_train, y_train, batch_size)
        # create tensor for training x and y
        batch_x = tf.Variable(batch_x)
        batch_y = tf.one_hot(tf.Variable(batch_y),10)

        with tf.GradientTape() as tape:
            logits = nn_model(batch_x, w1, b1, w2, b2)
            loss = loss_fn(logits, batch_y)
        
        gradients = tape.gradient(loss, [w1, b1, w2, b2])
        optimizer.apply_gradients(zip(gradients, [w1, b1, w2, b2]))
        avg_loss += loss/total_batch
    
    test_logits = nn_model(x_test, w1, b1, w2, b2)
    max_idx = tf.argmax(test_logits, axis = 1)
    test_acc = np.sum(max_idx.numpy() == y_test) / len(y_test)
    print(f"Epoch: {epoch + 1}, loss = {avg_loss:.3f}, test accuracy: {test_acc*100:.3f}%")

print("done!")
        
        

#%%
