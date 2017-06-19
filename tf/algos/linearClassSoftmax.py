import tensorflow as tf
import os
import time
import numpy as np
import sys
from tensorflow.python.client import timeline
sys.path.append(os.getcwd().rsplit("\\",1)[0])
import datasets.cifar10 as cifar


def main(argv):
    # Parameters defined
    N = 10 # Number of classes
    D = 3072 # Dimensions
    M = 32 # Minibatch size
    E = 10 # Num of epochs
    learning_rate = 0.0001
    # Import data
    trainData, trainLabels, testData, testLabels = cifar.load_CIFAR_10()
    
    

    # Preprocess
    trainData = trainData - np.mean(trainData)
    trainData = np.insert(trainData,trainData.shape[1],1,axis=1)
    testData = np.insert(testData,testData.shape[1],1,axis=1)
    oneHotTrainLabels = np.zeros([trainLabels.shape[0],10])
    oneHotTrainLabels[np.arange(trainLabels.shape[0]),trainLabels] = 1
    oneHotTestLabels = np.zeros([testLabels.shape[0],10])
    oneHotTestLabels[np.arange(testLabels.shape[0]),testLabels] = 1
    
    # Create Model
    x = tf.placeholder(tf.float32, [None, D+1])
    W = tf.Variable(0.1*tf.random_normal([D+1,N]))
    linear_model = tf.matmul(x,W)
    y = tf.placeholder(tf.float32, [None, N])

    # Define Loss
    # Loss is cross-entropy loss over all , defined by mean( -sum( y * log( linear_model ) ) )
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = linear_model))
    
    # Define Optimizer
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    train_step = tf.train.MomentumOptimizer(learning_rate,0.001,use_nesterov=True).minimize(cross_entropy)
    # Create and initialize session
    #sess = tf.InteractiveSession()
    #tf.global_variables_initializer().run()
    # Run the graph with full trace option
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        sess.run(train_step, {x: trainData[0:M], y: oneHotTrainLabels[0:M]}, options=run_options, run_metadata=run_metadata)

        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)

##    # Train
##    for _ in range(E):
##        for i in range(1,int(trainData.shape[0]/M)*M,M):
##            sess.run(train_step, {x: trainData[i:i+M], y: oneHotTrainLabels[i:i+M]})
##        print(sess.run(cross_entropy, {x: trainData, y: oneHotTrainLabels}))
##
##    # Test trained model
##    correctPrediction = tf.equal(tf.argmax(linear_model, 1), tf.argmax(y, 1))
##    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
##    print( sess.run( accuracy, {x: testData, y: oneHotTestLabels}))


if __name__ == '__main__':
    tf.app.run(main=main, argv=None)
