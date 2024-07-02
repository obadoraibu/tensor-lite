import numpy as np

class Dense():
    def __init__(self, input_size, output_size, activation_function, bias=True):
        self.weights = np.random.rand(input_size, output_size)
        self.bias = bias
        self.activation = activation_function
        
        if self.bias:
            self.biases = np.zeros(output_size)
        
    def forward_pass(self, x):
        result = np.dot(x, self.weights.T)
        if self.bias:
            result += self.biases
        return result
    
    def backward_pass():
        pass
    
    

class Conv():
    pass

class Polling():
    pass

class Dropout():
    pass

class BatchNorm():
    pass