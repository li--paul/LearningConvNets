import numpy as np
import time
import cifarload as cifar

strt = time.time()
print("Classification of CIFAR-10 dataset using regularized SVM with Minibatch gradient descent")

learning_rate = 0.0001
N = 10 # Number of classes
M = 64 # Size of minibatch
D = 3073 # Size of image
LAMBDA = 0.0001
#trainingSize = trainData.shape[0]
#testSize = testData.shape[0]
#w = np.zeros([N,D])
w = 0.01*np.random.randn(N,D)
r = []
t = []
# x (examples,3072)
# y (examples, 1)

def preprocess(x):
        x = x - np.mean(x)
        oldX = x
        sizeX = list(x.shape)
        sizeX[1] += 1
        x = np.ones(sizeX)
        x[:,:-1] = oldX
        return x
class ReLUneuron(object):
    def __init__(self,ins = D, outs = N):
        self.weights = 0.01*np.random.randn(outs,ins)
    def forward(self,inputs):
        self.out = np.maximum(0,np.dot(self.weights,inputs))
        return self.out
    def backprop(self,inputs,dout):
        # Backprop into ReLU function
        dout[self.out <= 0] = 0
        # Backprop into the weights
        dW = np.dot(dout,inputs) # xi must be transposed here, just like in evalgrad
        # Backprop into the inputs (for the prev. layer)
        dInp = np.dot(np.transpose(dout),self.weights)
        return dW, dInp
    def update(self,updateVal):
        self.weights += updateVal
    def getWeights(self):
        return self.weights

class outNeuron(object):
    def __init__(self,ins = N, outs = N):
        self.weights = 0.01*np.random.randn(outs,ins)
    def forward(self,inputs):
        self.out = np.dot(self.weights,inputs)
        return self.out
    def backprop(self, inputs, dout):
        # backprop into weights
        dW = np.dot(dout,inputs)
        # Backprop into inputs (for prev. layer)
        dInp = np.dot(np.transpose(dout),self.weights)
        return dW, dInp
    def update(self,updateVal):
        self.weights += updateVal
    def getWeights(self):
        return self.weights
def eval_grad(xi,yi,o,h1):
        global sca, xd, fl, yh
        delta = 1
        # linear combination
        hf1 = h1.forward(xi)
        f = o.forward(hf1)
        # grad for incorrect classes
        scaled = ((f-f[yi,range(M)]+delta)>0)*1
        #print(f-f[yi,range(M)])
        #time.sleep(1)
        # grad for correct class
        scaled[yi,range(M)] = -1*np.sum(scaled,0)
        scaled[yi,range(M)] += 1 # remove the contribution from the correct class
        grad = scaled/M
        #grad = np.dot(scaled,xi)
        #grad[:,:-1] += LAMBDA*W[:,:-1]
        #xd = grad
        dw2, din2 = o.backprop(np.transpose(hf1),grad)
        dw, din = h1.backprop(np.transpose(xi),np.transpose(din2))
        dw2[:,:-1] += LAMBDA*o.getWeights()[:,:-1]
        dw[:,:-1] += LAMBDA*h1.getWeights()[:,:-1]
        return dw, dw2
def train(o,h1):
        global w, sca
        [x,y,xt,yt] = cifar.load_CIFAR_10()
        x = preprocess(x)
        xt = preprocess(xt)
        for u in range(10):
                for i in range(1,int(x.shape[0]/M)*M,M):
                        dw, dw2 = eval_grad(np.transpose(x[i:i+M]),y[i:i+M],o,h1)
                        h1.update(-1*learning_rate*dw)
                        o.update(-1*learning_rate*dw2)
                print(i)
        return [xt,yt]
def test():
        global w, sca
        h1 = ReLUneuron(D,100)
        o = outNeuron(100,N)
        [xt,yt] = train(o,h1)
        sum1 = 0
        # Testing in Minibatches (increased performance vs Full batch)
        for i in range(1,int(xt.shape[0]/M)*M,M):
                res = o.forward(h1.forward(np.transpose(xt[i:i+M])))
                sum1 += (list(np.argmax(res,0)==yt[i:i+M]).count(True))
        print("Test accuracy is: %1.3f" % (sum1*100/(int(xt.shape[0]/M)*M))+" %")
        print("Total execution time: %1.3f s" % (time.time()-strt))
test()
