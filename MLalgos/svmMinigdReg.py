import numpy as np
import time
import cifarload as cifar
strt = time.time()
print("Classification of CIFAR-10 dataset using regularized SVM with Minibatch gradient descent")
#trainData,trainLabels,testData,testLabels = cifar.load_CIFAR_10()

learning_rate = 0.1
N = 10 # Number of classes
M = 32 # Size of minibatch
D = 3073 # Size of image
LAMBDA = 0.0001
#trainingSize = trainData.shape[0]
#testSize = testData.shape[0]
w = np.zeros([N,D])
r = []
t = []
# x (examples,3072)
# y (examples, 1)

def preprocess(x):
        x = x - 127
        oldX = x
        sizeX = list(x.shape)
        sizeX[1] += 1
        x = np.ones(sizeX)
        x[:,:-1] = oldX
        return x
##def Loss(xi,yi,W):
##        delta = 1
##        f = np.dot(W,xi)
##        diffs = np.maximum(0,(f - f[yi,range(M)] + delta))
##        diffs[yi] = 0
##        Lossi = diffs
##        return Lossi
def eval_grad(xi,yi,W):
        global sca,xd,fl,yh
        delta = 1
        # linear combination
        f = np.dot(W,xi)
        # grad for incorrect classes
        scaled = ((f-f[yi,range(M)]+delta)>0)*1
        # grad for correct class
        scaled[yi,range(M)] = -(np.sum(scaled,0)-1)
        grad = np.dot(scaled,np.transpose(xi))
        grad = grad + LAMBDA*W
        xd = xi
        return grad
def train():
        global w
        [x,y,xt,yt] = cifar.load_CIFAR_10()
        x = preprocess(x)
        xt = preprocess(xt)
        for i in range(1,int(x.shape[0]/M)*M,M):
                delw = eval_grad(np.transpose(x[i:i+M]),y[i:i+M],w)
                w -= learning_rate*delw
        
        return [xt,yt]

def test():
        global w
        [xt,yt] = train()
        sum1 = 0
        # Testing in Minibatches (increased performance vs Full batch)
        for i in range(1,int(xt.shape[0]/M)*M,M):
                res = np.dot(w,np.transpose(xt[i:i+M]))
                sum1 += (list(np.argmax(res,0)==yt[i:i+M]).count(True))
        print("Test accuracy is: %1.3f" % (sum1*100/9984)+" %")
        print("Total execution time: %1.3f s" % (time.time()-strt))
test()
