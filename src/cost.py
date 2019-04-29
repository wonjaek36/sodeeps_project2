import numpy as np

# For softmax
def cross_entropy(AL, Y):

    m = np.log(AL)
    m = np.nan_to_num(m)
    # print (m)
    #print (m.shape)
    cross_entropy = -np.sum(m*Y, axis=0, keepdims=True)
    
    return cross_entropy

def compute_cost(AL, Y):
    cet = cross_entropy(AL, Y)
    cost = np.mean(cet)
    assert(cost.shape) == ()

    return cost

def compute_cost_with_regularization(AL, Y, parameters, lambd):
    m = Y.shape[1]
    L = len(parameters) // 2 # number of parameters
    squared_weight_sum = 0
    for l in range(L):
        W = parameters["W" + str(l+1)]
        squared_weight_sum += np.sum(W**2)

    L2_regularization_cost = (squared_weight_sum * lambd) / (2 * m)
    return compute_cost(AL, Y) + L2_regularization_cost


"""
if __name__ == "__main__":

    Y = [
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [1.],
        [0.],
        [0.],
        [0.],
        [0.]
    ]

    X = [
        [1.91100668e-23],
        [1.49525814e-13],
        [1.22802545e-01],
        [6.09216600e-14],
        [2.32491442e-08],
        [9.69631345e-05],
        [8.64718290e-05],
        [3.03236871e-32],
        [2.58177784e-31],
        [8.77013997e-01]
    ]
    print (np.log(X)*Y)
    cost = cross_entropy(X, Y)
    print (cost)

"""