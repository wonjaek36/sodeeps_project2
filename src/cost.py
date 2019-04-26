import numpy as np

# For softmax
def cross_entropy(AL, Y):

    m = np.log(AL)
    m = np.nan_to_num(m)

    # print (m)
    #print (m.shape)
    cross_entropy = -np.sum(m*Y, axis=0)
    cost = np.mean(cross_entropy)
    #cost = np.squeeze(cost)

    assert(cost.shape) == ()

    return cost

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