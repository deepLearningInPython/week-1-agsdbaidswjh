import numpy

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.

# Task 1: 
# Instructions:
#Write a function that takes one numeric argument as input. 
#If the number is larger than zero, the function should return 1, otherwise is should return -1.
#The name of the function should be step

# Your code here:
# -----------------------------------------------

def step(num):
    if num > 0:
        return 1
    else:
        return -1

# -----------------------------------------------


# Task 2:
# Instructions:
#Write a function that takes in two arguments: a numpy array, and an integer (call argument "cutoff" and set default to 0).
#The function should return a numpy array of the same length, with all elements smaller than the cutoff being set to cutoff).
#The name of the function should be ReLu


# Your code here:
# -----------------------------------------------
def ReLu(vector, cutoff=0):
    result = vector.copy()
    result[result < cutoff] = cutoff
    return result

# -----------------------------------------------


# Task 3:
# Instructions:
#Write a function that takes in a two-dimensional numpy array of size (n, p) and a one-dimensional numpy array of size p.
#The function should start by multiplying the two numpy arrays (matrix multiplication).
#Next, apply the ReLu function from above to the resulting matrix and return the result.
#Name the function neural_net_layer

# Your code here:
# -----------------------------------------------

def neural_net_layer(matrix, vector):
    if matrix.shape[1] != vector.shape[0]:
        raise ValueError("Incompatible shapes for matrix multiplication.")
    
    result = numpy.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        product = 0
        for j in range(matrix.shape[1]):
            product += matrix[i][j] * vector[j]
        result[i] = product

    # or using numpy
    # result = numpy.dot(matrix, vector)

    return ReLu(result)

# ------------------------------------------