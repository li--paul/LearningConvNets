import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        ddict = pickle.load(fo, encoding='latin1')
    return ddict

def load_CIFAR_10():
    trainData = []
    trainLabels = []
    testData = []
    testLabels = []
    for i in range(1,6):
        rst = "../cifar-10-batches-py/data_batch_"+str(i)
        r = unpickle(rst)
        trainData.extend(r['data'])
        trainLabels.extend(r['labels'])
    rst = "../cifar-10-batches-py/test_batch"
    r = unpickle(rst)
    testData.extend(r['data'])
    testLabels.extend(r['labels'])
    return np.asarray(trainData),np.asarray(trainLabels),np.asarray(testData),np.asarray(testLabels)
