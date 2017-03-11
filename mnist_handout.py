from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import cPickle
import os
from scipy.io import loadmat


def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y), 0), (len(y), 1))


def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)


def forward(x, weights, biases):
    layers = [x]
    for i in range(len(weights)-1):
        layers.append(tanh_layer(layers[-1], weights[i], biases[i]))

    layers.append(linear_layer(layers[-1], weights[-1], biases[-1]))
    output = softmax(layers[-1])

    return layers[1:], output


def NLL(y, y_):
    return -sum(y_*log(y)) 


def deriv_multilayer(weights, biases, x, layers, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 = (y - y_)
    print("dCdL1", dCdL1.shape)
    dCdW1 = dot(layers[-2], dCdL1.T)
    dCdb1 = 1 * dCdL1
    print("dCdb1", dCdb1.shape)
    print("dCdW1", dCdW1.shape)
    dCdL0 = np.sum(weights[1] * dCdL1.T, 1).reshape(len(weights[1]), 1)
    print("dCdL0", dCdL0.shape)
    dL0dW0 = 1-layers[0]**2
    print(dL0dW0.shape)
    print(dCdL0.shape)
    dCdb0 = dL0dW0 * dCdL0
    dCdW0 = dot(x, (dL0dW0*dCdL0).T)
    print("dCdW0", dCdW0.shape)
    dCdx = np.sum(weights[0] * dCdL0.T, 1).reshape(len(weights[0]), 1)
    print("dCdx", dCdx.shape)
    return dCdL1, dCdW1, dCdb1, dCdL0, dCdW0, dCdb0, dCdx


def run_multilayer():
    # Load the MNIST digit data
    M = loadmat("data/mnist_all.mat")

    # Display the 150-th "5" digit from the training set
    #imshow(M["train5"][150].reshape((28, 28)), cmap=cm.gray)
    #show()

    # Load sample weights for the multilayer neural network
    snapshot = cPickle.load(open("data/snapshot50.pkl"))

    weights = []
    biases = []

    weights.append(snapshot["W0"])
    biases.append(snapshot["b0"].reshape((300, 1)))

    weights.append(snapshot["W1"])
    biases.append(snapshot["b1"].reshape((10, 1)))

    # Load one example from the training set, and run it through the
    # neural network
    x = M["train5"][148:149].T
    print(x.shape)
    layers, output = forward(x, weights, biases)
    # get the index at which the output is the largest
    y = np.array([0]*10).reshape(10,1)
    y[argmax(output)] = 1
    y_ = np.array([0]*10).reshape(10,1)
    y_[5] = 1

    print("Derivs")
    deriv_multilayer(weights, biases, x, layers, y, y_)


def linear_layer(prev_layer, W, b):
    """
    Create a linear layer
    :param prev_layer: 1-D array
    :param W: 2-D array with dimension 0 the size of prev_layer and dimension 1 is size of output
    :param b: 1-D array of size of output
    :return: output
    """
    return np.dot(W.T, prev_layer) + b


def run_linear():
    M = loadmat("data/mnist_all.mat")
    if os.path.exists("data/snapshot_linear.pkl"):
        snapshot = cPickle.load(open("data/snapshot_linear.pkl"))
        weights = [snapshot["W0"]]
        biases = [snapshot["b0"]]
    else:
        weights = [(np.random.randn(7840) * sqrt(2.0/7840)).reshape((784, 10))]
        biases = [np.ones(10).reshape((10,1))]

    x = M["train5"][148:149].T
    print(x.shape)
    layers, output = forward(x, weights, biases)
    outputs = softmax(layers[0])
    y = argmax(outputs)
    print(y)

if __name__ == "__main__":
    np.random.seed(7447)
    """mnist dataset has x of size 784"""
    #run_linear()
    run_multilayer()

################################################################################
#Code for displaying a feature from the weight matrix mW
#fig = figure(1)
#ax = fig.gca()    
#heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)    
#fig.colorbar(heatmap, shrink = 0.5, aspect=5)
#show()
################################################################################
