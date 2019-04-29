import numpy as np

def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}

    for l in range(L):
        (dW_row, dW_column) = parameters["W" + str(l+1)].shape
        (db_row, db_column) = parameters["b" + str(l+1)].shape
        v["dW" + str(l+1)] = np.zeros((dW_row, dW_column))
        v["db" + str(l+1)] = np.zeros((db_row, db_column))

    return v


def update_parameters_with_momentum(parameters, grads, v, learning_rate,  beta1):

    L = len(parameters) // 2

    for l in range(L):
        v["dW" + str(l+1)] = beta1*v["dW"+str(l+1)] + (1-beta1)*grads["dW"+str(l+1)]
        v["db" + str(l+1)] = beta1*v["db"+str(l+1)] + (1-beta1)*grads["db"+str(l+1)]

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v["dW"+str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v["db"+str(l+1)]

    return parameters, v


def initialize_RMSprops(parameters):

    L = len(parameters) // 2
    s = {}

    for l in range(L):
        (dW_row, dW_column) = parameters["W" + str(l+1)].shape
        (db_row, db_column) = parameters["b" + str(l+1)].shape
        s["dW" + str(l+1)] = np.zeros((dW_row, dW_column))
        s["db" + str(l+1)] = np.zeros((db_row, db_column))

    return s


def update_parameters_with_RMSprops(parameters, grads, s, learning_rate, beta2, epsillon):

    L = len(parameters) // 2

    for l in range(L):
        s["dW" + str(l+1)] = beta2*s["dW"+str(l+1)] + (1-beta2)*np.power(grads["dW"+str(l+1)], 2)
        s["db" + str(l+1)] = beta2*s["db"+str(l+1)] + (1-beta2)*np.power(grads["db"+str(l+1)], 2)

        # s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1-np.power(beta2, t))
        # s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1-np.power(beta2, t))

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW"+str(l+1)] / (np.sqrt(s["dW"+str(l+1)]) + epsillon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db"+str(l+1)] / (np.sqrt(s["db"+str(l+1)]) + epsillon)

    return parameters, s


def initialize_adam(parameters):

    L = len(parameters) // 2
    s = {}
    v = {}
    for l in range(L):
        (dW_row, dW_column) = parameters["W" + str(l+1)].shape
        (db_row, db_column) = parameters["b" + str(l+1)].shape
        v["dW" + str(l+1)] = np.zeros((dW_row, dW_column))
        v["db" + str(l+1)] = np.zeros((db_row, db_column))
        s["dW" + str(l+1)] = np.zeros((dW_row, dW_column))
        s["db" + str(l+1)] = np.zeros((db_row, db_column))

    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1=0.9, beta2=0.999, epsillon = 1e-8):

    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}
    for l in range(L):
        v["dW" + str(l+1)] = beta1*v["dW"+str(l+1)] + (1-beta1)*grads["dW"+str(l+1)]
        v["db" + str(l+1)] = beta1*v["db"+str(l+1)] + (1-beta1)*grads["db"+str(l+1)]

        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1-np.power(beta1, t))
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1-np.power(beta1, t))

        s["dW" + str(l+1)] = beta2*s["dW"+str(l+1)] + (1-beta2)*np.power(grads["dW"+str(l+1)], 2)
        s["db" + str(l+1)] = beta2*s["db"+str(l+1)] + (1-beta2)*np.power(grads["db"+str(l+1)], 2)

        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1-np.power(beta2, t))
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1-np.power(beta2, t))

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - (learning_rate * v_corrected["dW"+str(l+1)]) / (np.sqrt(s_corrected["dW"+str(l+1)]) + epsillon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - (learning_rate * v_corrected["db"+str(l+1)]) / (np.sqrt(s_corrected["db"+str(l+1)]) + epsillon)

    return parameters, v, s



def initialize_parameter(layers_dims):

    parameters = {}
    num_layer = len(layers_dims)
    for l in range(num_layer-1):
        parameters['W'+str(l+1)] = np.random.randn(layers_dims[l+1], layers_dims[l]) * 0.01
        parameters['b'+str(l+1)] = np.zeros((layers_dims[l+1], 1))


    return parameters


def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]

    return parameters