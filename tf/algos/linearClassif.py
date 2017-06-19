import tensorflow as tf
import os
import sys
os.chdir("../")
sys.path.append(os.getcwd())
import datasets.cifar10 as cifar

import numpy as np
import time
trainData,trainLabels,testData,testLabels = cifar.load_CIFAR_10()
# x_data = tf.placeholder(tf.float32, [None, 3072])
# y_data = tf.placeholder(tf.float32, [None])
os.chdir("./algos")
trainData = trainData - np.mean(trainData)
oneHotTrainLabels = np.zeros([trainLabels.shape[0],10])
oneHotTrainLabels[np.arange(trainLabels.shape[0]),trainLabels] = 1

# Model parameters
W = tf.Variable(tf.random_normal([3072, 10]), tf.float32)
b = tf.Variable(tf.random_normal([10]), tf.float32)
# Model input and output
x = tf.placeholder(tf.float32, [None, 3072])
linear_model = tf.matmul(x,W) + b
y = tf.placeholder(tf.float32, [None,10])
# loss
loss = tf.losses.hinge_loss(y, linear_model)
testCheck = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(y,axis=1),tf.argmax(linear_model,axis=1)),dtype=tf.float32))
#loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(10):
  sess.run(train, {x:trainData, y:oneHotTrainLabels})

# evaluate training accuracy
curr_W, curr_b, curr_loss, tc  = sess.run([W, b, loss, testCheck], {x:trainData, y:oneHotTrainLabels})
print("W: %s b: %s loss: %s %s"%(curr_W, curr_b, curr_loss, tc))
