import matplotlib.pyplot as pyplot
import numpy as np

f = open('E_3.txt', 'r')
train_acc = f.readline()
train_acc = train_acc[1:-2]
train_acc = list(map(float, train_acc.split(',')))

val_acc = f.readline()
val_acc = val_acc[1:-2]
val_acc = list(map(float, val_acc.split(',')))

cost = f.readline()
cost = cost[1:-2]
cost = list(map(float, cost.split(',')))

val_cost = f.readline()
val_cost = val_cost[1:-2]
val_cost = list(map(float, val_cost.split(',')))

pyplot.plot(train_acc)
pyplot.plot(val_acc)
pyplot.ylabel('accuracy')
pyplot.show()

pyplot.clf()

pyplot.plot(cost)
pyplot.plot(val_cost)
pyplot.ylabel('cost')
pyplot.show()