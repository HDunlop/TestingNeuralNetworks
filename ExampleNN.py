import sys
print(sys.version)


import numpy as np

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1.0-x)

class NeuralNetwork:
    def __init__(self,x,y):
        self.input = x
        self.param1 = np.random.rand(self.input.shape[1],4)
        self.param2 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def loop(self,iterations):
        for _ in range(iterations):
            self.feedForward()
            self.backPropagate()

    def feedForward(self):
        self.layer1 = sigmoid(np.dot(self.input,self.param1))
        self.output = sigmoid(np.dot(self.layer1,self.param2))

    def backPropagate(self):
        d_param2 = np.dot(self.layer1.T,(2*(self.y - self.output)*sigmoid_derivative(self.output)))
        d_param1 = np.dot(self.input.T,(np.dot(2*(self.y-self.output)*sigmoid_derivative(self.output),self.param2.T)*sigmoid_derivative(self.layer1)))

        self.param1 += d_param1
        self.param2 += d_param2

X = np.array(
    [[0,0,1],
    [0,1,1],
    [1,0,1]])
Y = np.array(
    [[0],
    [1],
    [1]])

nn = NeuralNetwork(X,Y)
nn.loop(1500)

print(nn.output)
