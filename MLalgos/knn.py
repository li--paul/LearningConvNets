import numpy as np
import time
import cifarload as cifar
strt = time.time()

print('k-Nearest Neighbour classification for "partial" CIFAR-10 dataset')
trainData,trainLabels,testData,testLabels = cifar.load_CIFAR_10()

class kNearestNeighbour(object):
    def __init__(self,l=1,k=1):
        # k = 1 is Nearest neighbour
        # l = 1 is L1 distance sum(abs(I1-I2))
        # l = 2 is L2 distance <<maybe>>sqrt(sum((I1-I2)^2))
        self.k = k
        self.l = l

    def train(self,X,Y):
        self.Xtrain = X
        self.Ytrain = Y

    def hyperParameters(self,l,k):
        self.l = l
        self.k = k
        
    def test(self,X):
        numRows = X.shape[0]
        Y = np.zeros(numRows,dtype = self.Ytrain.dtype)

        for i in range(numRows):
            if(self.l == 1):
                errs = np.sum(np.abs(self.Xtrain - X[i]),axis = 1)
            elif(self.l == 2):
                errs = np.sum(np.square(self.Xtrain - X[i]),axis = 1)
            # nearestImg = np.argmin(errs)
            # Y[i] = self.Ytrain[nearestImg]
            nearestImg = np.argsort(errs)[:self.k]
            Y[i] = np.argmax(np.bincount(self.Ytrain[nearestImg]))

        return Y
valac = []
kvals = [1, 3, 5, 10, 20, 50, 100]
validateData = trainData[1000:2000]
validateLabels = trainLabels[1000:2000]
# train
knn = kNearestNeighbour()
knn.train(trainData[0:1000],trainLabels[0:1000])
# validate
for k in kvals:
    knn.hyperParameters(1,k)
    Ytest = knn.test(validateData[0:1000])
    acc = (np.mean(Ytest == validateLabels[0:1000]))
    print('CV accuracy for '+str(k)+' nearest neighbour(s): %1.3f' % (acc*100)+"%")
    valac.append(acc)
# test
knn = kNearestNeighbour(1,kvals[np.argmax(np.asarray(valac))])
knn.train(trainData[0:1000],trainLabels[0:1000])
Ytest = knn.test(testData[0:1000])
acc = (np.mean(Ytest == testLabels[0:1000]))
print('\nTest accuracy: %1.3f' % (acc*100)+"%")

print('Total execution time: ' + str(time.time()-strt))
