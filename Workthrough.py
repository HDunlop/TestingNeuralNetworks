
import numpy as np
import matplotlib.pyplot as plt

# np.array bcos np requirement
input_features = np.array([[0,0],[0,1],[1,0],[1,1]])
target_output = np.array([0,1,1,1])

weights = np.array([0.1,0.2])

# sum of initial weights (constant) added to intermediate stage function
bias = 0.3

# distance from current point on plane to another
# e.g. big->risky but faster, small->cautious but slower
learning_rate = 0.05

def sigmoid(x):
    # return value between 0-1
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1.0-sigmoid(x))

# epoch traditionally used for generations
for epoch in range(5000):
    inputs = input_features
    # feeding forward
    ff = np.dot(inputs,weights)+bias
    # sigmoid output
    so = sigmoid(ff)
    # backpropagation
    error = so - target_output
    y = error.sum()

    plt.ion()
    if epoch % 200 == 0:
        plt.plot(epoch,y,'bo')
        plt.show()

    # calculate derivative of pathway
    # v derivative error to derivative output
    derror_doutput = error
    # v derivative so to derivative ff
    douto_dino = sigmoid_derivative(so)

    deriv = derror_doutput * douto_dino

    # currently linear so have to TRANSPOSE vector
    inputs = input_features.T
    deriv_final = np.dot(inputs,deriv)

    weights -= learning_rate * deriv_final

    # should you wish to adjust the bias, it is done as follows
    for i in deriv:
        bias -= learning_rate * i

        single_point = np.array([0,0])
        result1 = np.dot(single_point,weights) + bias
        result2 = sigmoid_derivative(result1)

        single_point = np.array([0,1])
        result1 = np.dot(single_point,weights) + bias
        result2 = sigmoid_derivative(result1)

        single_point = np.array([1,0])
        result1 = np.dot(single_point,weights) + bias
        result2 = sigmoid_derivative(result1)

        single_point = np.array([1,1])
        result1 = np.dot(single_point,weights) + bias
        result2 = sigmoid_derivative(result1)
