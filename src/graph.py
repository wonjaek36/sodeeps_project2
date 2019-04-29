import matplotlib.pyplot as pyplot
import numpy as np

f = open('B_3.txt', 'r')
train_acc = f.readline()
train_acc = train_acc[1:-2]
print (train_acc)
train_acc = list(map(float, train_acc.split(',')))
train_acc = train_acc[:-1]

val_acc = f.readline()
val_acc = val_acc[1:-2]
val_acc = list(map(float, val_acc.split(',')))
val_acc = val_acc[:-1]

cost = f.readline()
cost = cost[1:-2]
cost = list(map(float, cost.split(',')))
cost = cost[:-1]

pyplot.plot(train_acc)
pyplot.plot(val_acc)
pyplot.ylabel('accuracy')
pyplot.show()

pyplot.clf()

pyplot.plot(cost)
pyplot.ylabel('cost')
pyplot.show()