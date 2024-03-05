import numpy as np

# Output of 1 neuron with 3 inputs 
inputs = [1,2,3]
weights = [0.2,0.8,-0.5]
bias = 2
output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias

## Outputs of a layer composed of 3 neurons with 4 inputs each
inputs = [1,2,3,4]
weights1 = [0.2,0.8,-0.5, 1.0]
weights2 = [0.2,0.8,-0.5, 1.0]
weights3 = [0.2,0.8,-0.5, 1.0]

bias1 = 2.0
bias2 = 3.0
bias3 = 5.0

output = [
    inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + bias1, # output of neuron 1
    inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + bias2, # output of neuron 2
    inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + bias3] # output of neuron 3
          
## How to simplify this code ?
inputs = [1,2,3,4]
weights = [[0.2,0.8,-0.5, 1.0], # weights of neuron 1
           [0.2,0.8,-0.5, 1.0], # weights of neuron 2
           [0.2,0.8,-0.5, 1.0]] # weights of neuron 3
biases = [2.0,3.0,5.0]

layer_outputs = [] # Output of current layer
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0 # Output of given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += neuron_bias
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

## Simplify with vectors and dot product
# With a layer of 1 neuron
inputs = [1,2,3,4]
weights = [0.2,0.8,-0.5, 1.0]
bias = 2.0

output = np.dot(weights, inputs) + bias

# With a layer of 3 neurons and 4 inputs each
inputs = [1,2,3,4]
weights = [[0.2,0.8,-0.5, 1.0],
           [0.2,0.8,-0.5, 1.0],
           [0.2,0.8,-0.5, 1.0]]
biases = [2.0,3.0,5.0]

output = np.dot(weights, inputs) + biases # The first parameter of np.dot gives the index of the result
# For a better understanding : np.dot(weights, inputs) = [ np.dot(weights[0], inputs), np.dot(weights[1], inputs), np.dot(weights[2], inputs)]

## Batches
# Batches allow to calculate things in parallele, this is also the reason why we tend to use GPU instead of CPU (so much faster)
# Batches also help for generalization
# For a layer of 3 neurons with 4 features inputs each on a 3-sample batch 
inputs = [[1,2,3,4], # Feature of sample 1
          [1,2,3,4], # Feature of sample 2
          [1,2,3,4]] # Feature of sample 3
weights = [[0.2,0.8,-0.5, 1.0], # weights of neuron 1
           [0.2,0.8,-0.5, 1.0], # weights of neuron 2
           [0.2,0.8,-0.5, 1.0]] # weights of neuron 3
biases = [2.0,3.0,5.0]

outputs = np.dot(inputs, np.array(weights).T) + biases
# outputs = [[a, b, c]   -> output of neuron 1, 2 and 3 for the input 1
#            [d, e, f],  -> output of neuron 1, 2 and 3 for the input 2
#            [g, h, i]]  -> output of neuron 1, 2 and 3 for the input 3
# output + biases = output + [biases,
#                             biases,
#                             biases]

## Adding a layer 2
inputs = [[1,2,3,4],
          [1,2,3,4],
          [1,2,3,4]]
weights1 = [[0.2,0.8,-0.5, 1.0],
           [0.2,0.8,-0.5, 1.0],
           [0.2,0.8,-0.5, 1.0]]
weights2 = [[0.2,0.8,-0.5, 1.0],
           [0.2,0.8,-0.5, 1.0],
           [0.2,0.8,-0.5, 1.0]]
biases1 = [2.0,3.0,5.0]
biases2 = [2.0,3.0,5.0]

layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1
layer2_outputs = np.dot(inputs, np.array(weights2).T) + biases2

## Convert the concept of layer
X = [[1,2,3,4], # input data
     [1,2,3,4],
     [1,2,3,4]]
np.random.seed(0)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        #  np.random.randn does a Gaussian distribution bounded around 0
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons) # We multiply by 0.1 in order to have more often values between -1 and 1
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
# We try to minimize the range of weight initialisation and their values
# We want small values for weights because we hope things will continue to tend in a range of [-1,1] (normalized)-> to keep small value in sequential calculation due to layers
# Bias are usually initialized with 0, sometimes it is not fitting because a sum too small too activate the neuron will produce an output equals to 0 and propagate 0 trough the network

layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 12)
layer1.forward(X)
output = layer2.forward(layer1.output)

## Activation functions
