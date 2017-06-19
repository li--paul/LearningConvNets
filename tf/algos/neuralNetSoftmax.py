import tensorflow as tf
import os
import time
import numpy as np
import sys
from tensorflow.python.client import timeline
sys.path.append(os.getcwd().rsplit("\\",1)[0])
import datasets.cifar10 as cifar
import functools

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator

@doublewrap
def def_scope(function, scope=None, *args, **kwargs):
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class cifarStructure(object):
    def __init__(self):
        self.data = []
        self.labels = []
    
    def preprocess(self):
        self.data = self.data - np.mean(self.data)
        self.data = np.insert(self.data, self.data.shape[1], 1, axis=1)
        newLabels = np.zeros([self.labels.shape[0],10])
        newLabels[np.arange(self.labels.shape[0]), self.labels] = 1
        self.labels = newLabels

    def getShape(self):
        return list(self.data.shape)

class neuralNet(object):

    def __init__(self, N = 10, D = 3072, M = 32, E = 10, stepSize = 0.0001):
        self.train = cifarStructure()
        self.test = cifarStructure()
        self.classes = N
        self.dims = D
        self.minibatchSize = M
        self.epochs = E
        self.mom = 0.001
        self.learningRate = stepSize
        self.x = tf.placeholder(tf.float32, [None, self.dims+1])
        self.y = tf.placeholder(tf.float32, [None, self.classes])
        self.acc = tf.placeholder(tf.bool, [1])
        self.importData()
        self.preprocessData()
        self.model
        self.loss
        self.training
        self.train_acc
        self.run

    def importData(self):
        self.train.data, self.train.labels, self.test.data, self.test.labels = cifar.load_CIFAR_10()
        print(self.train.data)
        print("asdf")
        
    def preprocessData(self):
        self.train.preprocess()
        self.test.preprocess()

    @def_scope(initializer=tf.contrib.layers.xavier_initializer())
    def model(self):
        
        W = tf.Variable(0.1*tf.random_normal([self.dims+1,self.classes]))
        linearModel = tf.matmul(self.x,W)

        return linearModel

    @def_scope
    def loss(self):
        
        crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits = self.model))

        return crossEntropy

    @def_scope# (initializer=tf.contrib.layers.xavier_initializer())
    def training(self):
        mom2 = tf.train.MomentumOptimizer(self.learningRate, self.mom, use_nesterov=True)
        return mom2.minimize(self.loss)

    @def_scope
    def train_acc(self):
        correctPrediction = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_sum(tf.cast(correctPrediction, tf.float32))
        if self.acc == True:
            return tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
        else:
            return tf.reduce_sum(tf.cast(correctPrediction, tf.float32))

    @def_scope
    def run(self):
        with tf.Session() as sess:
            #x = tf.placeholder(tf.float32, [None, self.dims+1])
            #y = tf.placeholder((tf.float32, [None, self.classes])
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            runOptions = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            runMetadata = tf.RunMetadata()
            tf.shape(self.x)
            
            for _ in range(self.epochs):
                sumT = 0
                for minibatch in range(1,int(self.train.getShape()[0]/self.minibatchSize)*self.minibatchSize,self.minibatchSize):
                    [r,t] = sess.run([self.training, self.train_acc], {self.x: self.train.data[minibatch:minibatch+self.minibatchSize], self.y: self.train.labels[minibatch:minibatch+self.minibatchSize], self.acc: [False]}, options = runOptions, run_metadata = runMetadata)
                    #print(t/self.minibatchSize)
                    sumT += t
                print(sumT/self.train.getShape()[0])
            with open('timeline.json','w') as f:
                f.write(timeline.Timeline(runMetadata.step_stats).generate_chrome_trace_format())
            
n1 = neuralNet()


