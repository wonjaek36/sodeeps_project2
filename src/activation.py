import numpy as np
from cost import cross_entropy

def softmax(Z):
	cache = Z
	z = Z-np.max(Z, axis=0, keepdims=True)
	A = np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True, dtype=np.float64)[: np.newaxis]

	assert(A.shape == Z.shape)
	return A


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    assert(A.shape == Z.shape)

    return A, cache


def tanh(Z):
	"""
	Implements the hyperbolic tangent activation in numpy

	Arguments:
	Z -- numpy array of any shape

	Returns:
	A -- output of tanh(z), same shape as Z
	cache -- returns Z as well, useful during backpropagation
	"""
	E_2X = np.exp(Z) * np.exp(Z)
	A = (E_2X - 1) / (E_2X + 1) 
	cache = Z
	
	assert(A.shape == Z.shape)

	return A, cache


def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def leaky_relu(Z):
	"""
	Implements the Leaky RELU function.

	Arguments:
	Z -- numpy array of any shape

	Returns:
	A -- Post activation(leaky relu) parameters, of the same shape as Z
	cache -- returns Z as well, useful during back propagation
	"""

	A = np.maximum(0.01*Z, Z)
	assert(A.shape == Z.shape)

	cache = Z
	return A, cache


"""
def softmax_backward(dA, cache):
	Z = cache
	z = Z-np.max(Z, axis=0)
	s = np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True, dtype=np.float64)[: np.newaxis]
	dZ = dA * s * (1-s)

	assert (dZ.shape == Z.shape)

	return dZ
"""
	

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ



def tanh_backward(dA, cache):
	"""
	Implement the backward propagation for hyperbolic tangent function.

	Arguments:
	dA -- post-activation gradient, of any shape
	cache --- 'Z' where we store for computing backward propagation efficiently

	Returns:
	dZ -- Gradient of the cost with respect to Z
	"""
	Z = cache

	E_2X = np.exp(Z) * np.exp(Z)
	dZ = dA * (4 / (E_2X + 2 + (1/E_2X)))
	
	assert (dZ.shape == Z.shape)

	return dZ


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def leaky_relu_backward(dA, cache):
	"""
	Implement the backward propagation for a leaky ReLU unit.

	Arguments:
	dA -- post-activation gradient, of any shape
	cache -- 'Z' where we store for computing backward propagation efficiently.

	Returns:
	dZ -- Gradient of the cost with respect to Z
	"""

	Z = cache
	dZ = np.array(dA, copy=True)

	# When z <= 0, you should multiply dz to 0.01.
	dZ[Z <= 0] *= 0.01

	assert (dZ.shape == Z.shape)

	return dZ


if __name__ == "__main__":
	Z = [
		[2,-0.99], #,0.3,0.1]#,
		[2,-0.99], #,3,4],
		[2,0.99] #,4,5]
	]
	Y = [
	 [1,0],
	 [0,0],
	 [0,1]# ,1,1]#,
	 #[0,0,0],
	 #[0,0,0]
	]
	Z = np.asarray(Z)
	print ( np.sum(Z, axis=0, keepdims=True, dtype=np.float) )
	A = softmax(Z)
	print (A)
	cost = cross_entropy(A, Y)
	print (cost)

	dA = softmax_backward(A, Y)
	print (dA)