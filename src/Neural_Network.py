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
        self.activation = config['MODEL']['activation_functions']

        layers_dims = [n_x]
        layers_dims.extend(n_h)
        layers_dims.append(n_y)

        print (layers_dims)
        self.initialize_parameter(layers_dims)
        logging.info('Completing initialize parameter')


    def initialize_parameter(self, layers_dims):

        parameters = {}
        num_layer = len(layers_dims)
        for l in range(num_layer-1):
            parameters['W'+str(l+1)] = np.random.randn(layers_dims[l+1], layers_dims[l]) * 0.01
            parameters['b'+str(l+1)] = np.zeros((layers_dims[l+1], 1))
            print (parameters['W'+str(l+1)].shape)

        self.parameters = parameters


    def train_model(self):
        
        epoch = config['TRAIN']['epoch']
        batch_size = config['TRAIN']['batch_size']
        train_data_ratio = config['TRAIN']['train_data_ratio']
        validation_data_ratio = config['TRAIN']['validation_data_ratio']
        learning_rate = config['TRAIN']['learning_rate']

        numTrain = int(self.trainX.shape[1]*train_data_ratio)
        numVal = int(self.trainX.shape[1] - numTrain)

        print (numTrain)
        print (numVal)
        trainX = self.trainX[:, 0:numTrain]
        trainY = self.trainY[:, 0:numTrain]
        val_X = self.trainX[:, numTrain: ]
        val_Y = self.trainY[:, numTrain: ]

        numBatch = numTrain // batch_size

        #print (trainX.shape)
        #print (val_X.shape)
        #print (numTrain)
        #print ('batch_size: ' + str(batch_size))
        #print (numBatch)
        for i in range(epoch):
            for j in range(numBatch):
                batch_X = trainX[:, j*batch_size:(j+1)*batch_size]
                batch_Y = trainY[:, j*batch_size:(j+1)*batch_size]

                AL, caches = self.forward_propagation(batch_X)
                cost = self.compute_cost(AL, batch_Y, 'cross_entropy')
                print ('Epoch ' + str(i) + ' - ' + 'cost ' + str(cost))

                grads = self.backward_propagation(AL, batch_Y, caches)
                self.parameters = self.update_parameters(self.parameters, grads, learning_rate)
                train_acc, val_acc = self.estimate(AL, batch_Y, val_X, val_Y)
                print ('train_accuracy: ' + str(train_acc))
                if val_acc is not None:
                    print ('val_accuracy: ' + str(val_acc))

            # Last batch
            if numTrain % batch_size != 0:
                batch_X = trainX[:, numBatch*batch_size:]
                batch_Y = trainY[:, numBatch*batch_size:]
                print (batch_X.shape)
                AL, caches = self.forward_propagation(batch_X)
                cost = self.compute_cost(AL, batch_Y, 'cross_entropy')
                print ('Epoch ' + str(i) + ' - ' + 'cost ' + str(cost))

                grads = self.backward_propagation(AL, batch_Y, caches)
                self.parameters = self.update_parameters(self.parameters, grads, learning_rate)
                train_acc, val_acc = self.estimate(AL, batch_Y, val_X, val_Y)
                print ('train_accuracy: ' + str(train_acc))
                if val_acc is not None:
                    print ('val_accuracy: ' + str(val_acc))


            # train_acc, val_acc = self.estimate(AL, batch_Y, val_X, val_Y)
            #print ('train_accuracy: ' + str(train_acc))
            #if val_acc is not None:
            #    print ('val_accuracy: ' + str(val_acc))

            # break
            # W1 = self.parameters["W1"]
            # print (W1.shape)
            # print (W1[0].shape)


    def forward_propagation(self, batch_X):

        parameters = self.parameters
        num_layers = len(parameters) // 2
        A = batch_X
        caches = []
        for i in range(1, num_layers):
            # linear forward
            W = parameters['W'+str(i)]
            b = parameters['b'+str(i)]
            A, cache = self.activation_forward(A, W, b, self.activation[i-1])
            caches.append(cache)

        # Output Layer
        W = parameters['W'+str(num_layers)]
        b = parameters['b'+str(num_layers)]
        AL, cache = self.activation_forward(A, W, b, self.activation[-1])
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

        elif activation == "tanh":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = tanh(Z)

        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)

        elif activation == "leaky_relu":
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
            grads["dW" + str(l+1)] = dW
            grads["db" + str(l+1)] = db

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


    def prediction(self ):
        pass


    def estimate(self, AL, Y, val, val_Y):

        numData = Y.shape[1]
        prediction = np.argmax(AL, axis=0)
        solution = np.argmax(Y, axis=0)

        right = np.sum(prediction == solution)
        train_accuracy = right / numData

        val_accuracy = None
        if val is not None and val_Y is not None:
            val_AL, _ = self.forward_propagation(val)
            # cost = self.compute_cost(val_AL, val_Y, 'cross_entropy')

            val_pred = np.argmax(val_AL, axis=0)
            val_sol = np.argmax(val_Y, axis=0)

            val_right = np.sum(val_pred == val_sol)
            val_accuracy = val_right / val_Y.shape[1]


        return train_accuracy, val_accuracy



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

