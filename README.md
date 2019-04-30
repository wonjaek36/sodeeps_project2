# sodeeps_project2

## Introduction
Sodeeps Project Homework 2  
MNIST data is used for testing Standard Neural Network algorithm.  
[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

## Data
ubyte data should be in /resources folder  
read_mnist.py will read data from that directory  

## Configuration
This nerual network is configurable through config.json  
Especially, hyper-parameter in below is configurable in that file.  

## Hyper-Parameter

* Number of Hidden layer
* Number of Neuron in each layer
* Activation Function
  * Relu
  * Leaky Relu
  * Sigmoid
  * tanh
* Mini-Batch size
* Regularization
  * L2 Regularization
  * L1 Regularization(Future work)
  * Drop-out(Future work)
* Optimizer
  * Gradient Descent
  * Momentum(beta)
  * RMSprops(beta, epsilon)
  * Adam(beta1, beta2, epsilon)
* Ratio of train/validation/test dataset
