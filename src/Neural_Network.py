from read_mnist import DataContainer
import numpy as np
import sys
import os
from array import array
import json
import logging


from activation import softmax, softmax_backward, sigmoid, sigmoid_backward, tanh, tanh_backward, relu, relu_backward, leaky_relu, leaky_relu_backward
from cost import cross_entropy

class Neural_Network():

    def __init__(self, config):
        self.config = config
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    def read_mnist(self):
        resource_path = config['DEFAULT']['RESOURCE_PATH']
        dc = DataContainer(resource_path)
        trainX, trainY = dc.read_mnist()

        self.trainX = trainX
        self.trainY = trainY

        logging.info('Read mnist data complete')


    def build_model(self):

        # get model information from config
        n_x = config['MODEL']['features']
        n_h = config['MODEL']['hidden_layers']
        n_y = config['MODEL']['labels']
        layers_dims = [n_x]
        layers_dims.extend(n_h)
        layers_dims.append(n_y)

        self.initialize_parameter(layers_dims)
        logging.info('Completing initialize parameter')


    def initialize_parameter(self, layers_dims):

        parameters = {}
        num_layer = len(layers_dims)
        for l in range(num_layer-1):
            parameters['W'+str(l+1)] = np.random.randn(layers_dims[l+1], layers_dims[l]) * 0.01
            parameters['b'+str(l+1)] = np.zeros((layers_dims[l+1], 1))

        self.parameters = parameters


    def train_model(self):
        
        epoch = config['TRAIN']['epoch']
        batch_size = config['TRAIN']['batch_size']


        batch_X = self.trainX[:, 0:3]
        batch_Y = self.trainY[:, 0:3]
        for i in range(epoch):
            # TODO Mini-batch
            # ------

            AL, caches = self.forward_propagation(batch_X)
            cost = self.compute_cost(AL, batch_Y, 'cross_entropy')
            print ('Epoch ' + str(i) + ' - ' + 'cost ' + str(cost))

            grads = self.backward_propagation(AL, batch_Y, caches)
        
            self.parameters = self.update_parameters(self.parameters, grads, 0.0001)

            W1 = self.parameters["W1"]
            #print (W1.shape)
            #print (W1[0].shape)


    def forward_propagation(self, batch_X):

        parameters = self.parameters
        num_layers = len(parameters) // 2
        A = batch_X
        caches = []
        for i in range(1, num_layers):
            # linear forward
            W = parameters['W'+str(l)]
            b = parameters['b'+str(l)]
            A, cache = self.activation_forward(A, W, b, 'relu')
            caches.append(cache)

        # Output Layer
        W = parameters['W'+str(num_layers)]
        b = parameters['b'+str(num_layers)]
        AL, cache = self.activation_forward(A, W, b, 'softmax')
        caches.append(cache)


        #assert (AL.shape == (1, X.shape[1]))

        return AL, caches


    def linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b

        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        return Z, cache


    def activation_forward(self, A_prev, W, b, activation):

        if activation == 'softmax':
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = softmax(Z)

        
        elif activation == 'sigmoid':
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)

        elif activation_relu == "tanh":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = tanh(Z)

        elif activation_relu == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)

        elif activation_relu == "leaky_relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = leaky_relu(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache


    def compute_cost(self, AL, Y, func):
        if func == 'cross_entropy':
            return cross_entropy(AL, Y)
        else:
            pass

    def backward_propagation(self, AL, Y, caches):
        
        grads = {}
        L = len(caches)

        m = AL.shape[1]
        # Y = Y.reshape(AL.shape) 

        # Initializing the backward propagation
        dAL = AL - Y # Softmax

        cache = caches[-1]
        grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)] = self.linear_activation_backward(dAL, cache, "softmax")
        for l in reversed(range(L-1)):
            cache = caches[l]
            dA_prev, dW, db = self.linear_activation_backward(grads["dA"+str(l+1)], cache, "relu")
            grads["dA" + str(l)] = dA_prev
            grads["dW" + str(l)] = dW
            grads["db" + str(l)] = db

        return grads


    def linear_backward(self, dZ, cache):

        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db


    def linear_activation_backward(self, dA, cache, activation):

        linear_cache, activation_cache = cache
        if activation == "softmax":
            dZ = softmax_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation == "relu":
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation == "leaky_relu":
            dZ = leaky_relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation == "tanh":
            dZ = tanh_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)    

        return dA_prev, dW, db

    def update_parameters(self, parameters, grads, learning_rate):

        L = len(parameters) // 2

        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]

        return parameters

if __name__ == '__main__':

    config = None
    with open(os.path.join('src', 'config.json'), 'r') as f:
        config = json.load(f)

    if config is not None:
        nn = Neural_Network(config)
        nn.read_mnist()

        nn.build_model()
        nn.train_model()
        # lr.prediction()

