import numpy as np
import sys
import os
from array import array
import json

from struct import unpack
import matplotlib
import matplotlib.pyplot as pyplot
import matplotlib.cm as cm

class DataContainer():

    def __init__(self, resource_path):
        self.resource_path = resource_path
        self.trainX_filename = 'train-images.idx3-ubyte'
        self.trainY_filename = 'train-labels.idx1-ubyte'
        self.testX_filename  = 't10k-images.idx3-ubyte'
        self.testY_filename  = 't10k-images.idx1-ubyte'


    def read_mnist(self):
        # Only for using training example for now
        #TODO using whole data(train, test)

        images = open(os.path.join(self.resource_path, self.trainX_filename), 'rb')
        labels = open(os.path.join(self.resource_path, self.trainY_filename), 'rb')
        self.images = images
        self.labels = labels

        (_, numX, rowX, rowY) = unpack('>iiii', images.read(16))
        (_, numY) = unpack('>ii', labels.read(8))

        self.numX = numX
        self.numY = numY
        self.rowX = rowX
        self.rowY = rowY

        trainX = []
        trainY = []

        for i in range(numX):
            s = images.read(rowX * rowY)
            l = labels.read(1)
            image = unpack(len(s)*'B', s)
            label = int(unpack('B', l)[0])
            label_onehot = np.zeros(10)
            label_onehot[label] = 1

            trainX.append(image)
            trainY.append(label_onehot)

        trainX = np.asarray(trainX)
        trainY = np.asarray(trainY)
        # print (trainY[0])
        # trainY = trainY.reshape((self.numY, 1))

        trainX = np.swapaxes(trainX, 0, 1)
        trainY = np.swapaxes(trainY, 0, 1)

        # Normalize data
        # trainX = trainX / 255.0
        # print (trainY.shape)
        return trainX, trainY
        

    def show_first_example(self):

        images = self.images
        labels = self.labels
        images.seek(16)
        labels.seek(8)

        s = images.read(784)
        l = labels.read(1)

        # TODO studying about upack
        image = np.reshape( unpack(len(s)*'B', s), (28, 28))
        label = unpack('B', l)[0]
        # print (label)
        print (int(label))
        pyplot.imshow(image, cmap='binary')
        pyplot.show()


if __name__ == '__main__':

    config = None
    with open(os.path.join('src', 'config.json'), 'r') as f:
        config = json.load(f)

    if config is not None:
        resource_path = config['DEFAULT']['RESOURCE_PATH']
        dc = DataContainer(resource_path)
        dc.read_mnist()
        dc.show_first_example()