import numpy as np



#        N1 N2 N3
inputs = [[1,2,3],#input1
         [2,2,3],#input2
         [4,5,2]]#input3

#         N1 N2 N3
weights = [[1,1,1],# output node 1
           [2,2,2]]# output node 2

#       N1 N2 N3
biases = [1,2]

output = np.dot(inputs, np.array(weights).T)+biases
print(output)

