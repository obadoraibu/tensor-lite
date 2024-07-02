import numpy as np

class LTensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._grad_fn = None 

    def set_fn(self, function):
        self._grad_fn = function

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data, dtype=np.float32)
        
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
        
        if self._grad_fn is not None:
            self._grad_fn.backward(grad)
    
    def __add__(self, other):
        return Add.apply(self, other)
    
    def __mul__(self, other):
        return Mul.apply(self, other)
      
class Function:
    @classmethod
    def apply(cls, *inputs):
        obj = cls()
        obj.inputs = inputs
        obj.output = obj.forward(*inputs)
        obj.output.set_fn(obj)
        return obj.output
        
    def forward(self, *inputs):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError
   
class Add(Function):
    def forward(self, a: LTensor, b: LTensor):
        self.a = a
        self.b = b
        return LTensor(a.data + b.data, requires_grad=(a.requires_grad or b.requires_grad))

    def backward(self, grad):
        if self.a.requires_grad:
            self.a.backward(grad)
        if self.b.requires_grad:
            self.b.backward(grad)

class Mul(Function):
    def forward(self, a: LTensor, b: LTensor):
        self.a = a
        self.b = b
        return LTensor(a.data * b.data, requires_grad=(a.requires_grad or b.requires_grad))

    def backward(self, grad):
        if self.a.requires_grad:
            self.a.backward(grad * self.b.data)
        if self.b.requires_grad:
            self.b.backward(grad * self.a.data)



def test_autograd():
    # Create tensors
    a = LTensor([2.0], requires_grad=True)
    b = LTensor([3.0], requires_grad=True)
    
    # Perform operations
    c = a + b  # c = a + b
    d = a * b  # d = a * b
    e = c + d  # e = c + d
    
    # Perform backward pass
    e.backward()
    
    # Check values
    assert np.allclose(e.data, [11.0]), f"Expected e.data to be [11.0], but got {e.data}"
    assert np.allclose(a.grad, [4.0]), f"Expected a.grad to be [4.0], but got {a.grad}"
    assert np.allclose(b.grad, [3.0]), f"Expected b.grad to be [3.0], but got {b.grad}"
    
    print("Test passed!")


# Run the test
test_autograd()