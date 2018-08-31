
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

seed = 7
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()

print ("Training Instances", train.shape[0])
print ("Training Instances", test.shape[0])
print ("Atrributes", test.shape[1])
y_train = train.pop('label')

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train, y_train, 
                                                    stratify=y_train,
                                                    random_state=seed)
def data_transform(data, labels):
    data = data.astype(np.float32)
    if labels is not None:
        labels = (np.arange(10) == labels[:,None]).astype(np.float32)
    return data, labels

X_train, y_train = data_transform(X_train.values, y_train)
X_val, y_val = data_transform(X_val.values, y_val)
X_test, _ = data_transform(test.values, None)

print ("Training dataset dimensions=",X_train.shape, "\tTraining labels=",y_train.shape)
print ("Validation dataset dimensions=",X_val.shape, "\tValidation labels=",y_val.shape)
print ("Testing Dataset dimensions=", X_test.shape)

import tensorflow as tf

tf.reset_default_graph()
K, L, M, N = 200, 100, 60, 30

# input
X = tf.placeholder(tf.float32, [None, 28*28])

# weights and biases of 5 fully connected layers
w1 = tf.Variable(tf.truncated_normal([28*28, K], stddev=0.1))
b1 = tf.Variable(tf.zeros([K]))

w2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
b2 = tf.Variable(tf.zeros([L]))

w3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
b3 = tf.Variable(tf.zeros([M]))

w4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
b4 = tf.Variable(tf.zeros([N]))

w5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
b5 = tf.Variable(tf.zeros([10]))

y1 = tf.nn.relu(tf.matmul(X, w1)+b1)
y2 = tf.nn.relu(tf.matmul(y1, w2)+ b2)
y3 = tf.nn.relu(tf.matmul(y2, w3)+ b3)
y4 = tf.nn.relu(tf.matmul(y3, w4)+ b4)

Y = tf.matmul(y4, w5)+ b5
Y_true = tf.placeholder(tf.float32, [None, 10])

loss = tf.nn.softmax_cross_entropy_with_logits(logits=Y, labels=Y_true)
mean_loss = tf.reduce_mean(loss)
is_correct = tf.equal(tf.argmax(Y, 1),tf.argmax(Y_true,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.003
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           5000, 0.75, staircase=True)

optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss, global_step=global_step)

sess = tf.InteractiveSession()
initializer = tf.global_variables_initializer()
sess.run(initializer)

batch_size = 100
batch_number = X_train.shape[0]//batch_size

for epoch_counter in range(100):
    curr_epoch_loss = 0
    start = 0
    end = start + batch_size
    
    # training the network on batches
    for batch_counter in range(batch_number):
        batch_x = X_train[start:end]
        batch_y = y_train[start:end]
        start = end
        end = start+batch_size
        
        train_data = {X: batch_x, Y_true: batch_y}
        _, batch_loss = sess.run([optimize,mean_loss], feed_dict=train_data)
        curr_epoch_loss += batch_loss
    
    curr_epoch_loss /= batch_number   
    val_data = {X: X_val, Y_true: y_val}
    val_loss, val_accuracy = sess.run([mean_loss,accuracy], feed_dict=val_data)
    
    print ("Epoch %d: Train Loss=%0.4f Val Loss=%0.4f Val Acc=%0.4f eta=%0.6f global_step=%d"
          % (epoch_counter+1, 
             curr_epoch_loss, 
             val_loss, 
             val_accuracy, 
             learning_rate.eval(session=sess),
             global_step.eval(session=sess)))
predict = tf.argmax(tf.nn.softmax(Y),1)
predictions = predict.eval(feed_dict={X: X_test})

test_id = np.arange(1, len(X_test)+1)
submission = pd.DataFrame({'ImageId': test_id, 'Label':predictions})
submission.head()
submission.to_csv('my_submission.csv',index=False)

