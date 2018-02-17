
# coding: utf-8

# In[1]:

import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib import learn
from data_utils import *
embedding_size = 128
filter_sizes =[3,4]
num_filters = 32
num_classes =20
batch_size = 64
num_epochs = 10


# In[2]:

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# In[3]:

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# In[4]:

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


# In[5]:

def max_pool(x, ksize):
    return tf.nn.max_pool(x, ksize,strides=[1, 1, 1, 1], padding='VALID')


# In[6]:

x_train, y_train, x_test, y_test = load_data_and_labels()
#x_train=x_trai[0:1000]
#y_train=y_trai[0:1000]
print("Number of training example",len(x_train))
max_document_length = max([len(x.split(" ")) for x in x_train])
print("maximum length of doc",max_document_length)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x_train = np.array(list(vocab_processor.fit_transform(x_train)))
x_test = np.array(list(vocab_processor.fit_transform(x_test)))
sequence_length=x_train.shape[1]
num_classes=y_train.shape[1]
vocab_size=len(vocab_processor.vocabulary_)
print("number of classes",num_classes)


# In[7]:

input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")


# In[8]:

W = weight_variable([vocab_size, embedding_size])
embedded_chars = tf.nn.embedding_lookup(W,input_x)
embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)


# In[9]:

pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    # Convolution Layer
    filter_shape = [filter_size, embedding_size, 1, num_filters]
    W = weight_variable(filter_shape)
    b = bias_variable([num_filters])
    conv = conv2d(embedded_chars_expanded, W)
    # Apply nonlinearity
    h = tf.nn.relu(tf.nn.bias_add(conv, b))
    # Maxpooling over the outputs
    pooled = max_pool(h, [1, sequence_length - filter_size + 1, 1, 1])
    pooled_outputs.append(pooled)
#print(len(pooled_outputs))    
num_filters_total = 32 * len(filter_sizes)
#print(num_filters_total)
h_pool = tf.concat(pooled_outputs,2)
#print(h_pool.shape)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
W = weight_variable([num_filters_total, num_classes])
b = bias_variable([num_classes])
scores = tf.nn.softmax(tf.matmul(h_pool_flat, W) + b)
#print(scores.shape)
pred = tf.nn.softmax(scores)
predictions = tf.argmax(scores, 1)


# In[10]:

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(                    logits=scores, labels=input_y))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
train_op = optimizer.minimize(loss_op)
correct_predictions = tf.equal(predictions, tf.argmax(input_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))


# In[11]:

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    print(data.shape)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# In[12]:

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# In[ ]:

print(x_train.shape)
print(y_train.shape)
print(len(np.array(list(zip(x_train, y_train)))))
batches = batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs)
for batch in batches:
    x_batch, y_batch = list(zip(*batch))
    _, cost = sess.run([train_op,loss_op], feed_dict={input_x: x_batch,input_y: y_batch})
    print( "Loss: {}".format(cost))
    test_accuracy = sess.run(accuracy,feed_dict={input_x: x_batch,input_y: y_batch})
    #print("test accuracy: {}".format(test_accuracy))


# In[14]:

print(x_test.shape)
print(y_test.shape)
sum1=0
num1=0
print(len(np.array(list(zip(x_test, y_test)))))
batches = batch_iter(list(zip(x_test, y_test)), batch_size, 1)
for batch in batches:
    num1=num1+1
    x_batch, y_batch = list(zip(*batch))
    test_accuracy = sess.run(accuracy,feed_dict={input_x: x_batch,input_y: y_batch})
    #print("test accuracy: {}".format(test_accuracy))
    sum1=sum1+test_accuracy


# In[15]:

print(num1)
print(sum1)
total1=sum1*100/num1
print("Accuracy for test data-",total1)


# In[16]:

print(x_train.shape)
print(y_train.shape)
sum2=0
num2=0
print(len(np.array(list(zip(x_train, y_train)))))
batches = batch_iter(list(zip(x_train, y_train)), batch_size, 1)
for batch in batches:
    num2=num2+1
    x_batch, y_batch = list(zip(*batch))
    train_accuracy = sess.run(accuracy,feed_dict={input_x: x_batch,input_y: y_batch})
    #print("train accuracy: {}".format(train_accuracy))
    sum2=sum2+train_accuracy


# In[17]:

print(num2)
print(sum2)
total2=sum2*100/num2
print("Accuracy for train data-",total2)

