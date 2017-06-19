import pickle
import numpy as np
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        ddict = pickle.load(fo, encoding='latin1')
    return ddict

def load_CIFAR_10():
    trainData = []
    trainLabels = []
    testData = []
    testLabels = []
    pathVal = os.path.dirname(__file__).rsplit("\\", 2)[0] + "\\cifar-10-batches-py\\"
    for i in range(1,6):
        r = unpickle(pathVal+"data_batch_"+str(i))
        trainData.extend(r['data'])
        trainLabels.extend(r['labels'])
    r = unpickle(pathVal+"test_batch")
    testData.extend(r['data'])
    testLabels.extend(r['labels'])
    return np.asarray(trainData),np.asarray(trainLabels),np.asarray(testData),np.asarray(testLabels)
