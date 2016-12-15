# 2 layer neural network to predict XOR value of some data
import numpy as np
import time

# hyperparameters

# Number of neurons per layer
n_hidden = 10
n_in = 10
n_out = 10

n_samples = 300

# These hyperparameters are used to lower training loss in 
# our NN using cross entropy
learning_rate = 0.01
momentum = 0.9

# Non-deterministic seeding
np.random.seed(0)

# Sigmoid activation function (turns numbers into probabilities)
def sigmoid(x): 
	return 1.0/(1.0 + np.exp(-x))

def tanh_prime(x):
	return 1.0 - np.tanh(x)**2

# input, transpose, 1st layer weights, 2nd layer weights, biases
def train(x, t, V, W, bv, bw):

	# forward prop-- matrix multiply + biases
	A = np.dot(x, V) + bv
	Z = np.tanh(A) # Computing the result of the dot product through an activation function creates the delta

	B = np.dot(Z, W) + bw
	Y = sigmoid(B)

	# backward prop
	# Find two errors
	Ew = Y - t # transpose is a matrix of weights, flipped. We flip it because we are now doing backprop
	Ev = tanh_prime(A) * np.dot(W, Ew) # Ev is used to predict loss

	# predict our loss 
	# Find two deltas
	dW = np.outer(Z, Ew)
	dV = np.outer(x, Ev)

	# Cross-entropy loss function (many many others exist, like mean MSE, etc)
	# Cross-entropy is great for classification tasks
	loss = -np.mean(t * np.log(Y) + (1 - t) * np.log(1 - Y))

	return loss, (dV, dW, Ev, Ew) # return loss and gradient

def predict(x, V, W, bv, bw):
	A = np.dot(x, V) + bv
	B = np.dot(np.tanh(A), W) + bw
	return (sigmoid(B) > 0.5).astype(int)


# create layers
V = np.random.normal(scale=0.1, size=(n_in, n_hidden))
W = np.random.normal(scale=0.1, size=(n_hidden, n_out))

bv = np.zeros(n_hidden)
bw = np.zeros(n_out)

params = [V, W, bv, bw]

# generate our data
X = np.random.binomial(1, 0.5, (n_samples, n_in))
T = X ^ 1 # generate XORed

# Training time
for epoch in range(100):
	
	err = []
	update = [0] * len(params)
	t0 = time.clock()

	# for each sample, update weights in network
	for i in range(X.shape[0]):
		loss, grad = train(X[i], T[i], *params)
		
		for j in range(len(params)):
			params[j] -= update[j]

		for j in range(len(params)):
			update[j] = learning_rate * grad[j] + momentum * update[j]

		err.append(loss)
		print('Epoch: %d, Loss: %.8f, Time: %.4fs' % (epoch, np.mean(err), time.clock() - t0))

# Try to predict something
x = np.random.binomial(1, 0.5, n_in)
print('XOR prediction | input, output, ground-truth')
print(x)
print(predict(x, *params))
print(x ^ 1)
