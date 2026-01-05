# Goal is to create mini version of transformer architecture
# scope will be numpy only

### System design
# Layers- Linear, Norm, Attention head, 
# Combo - Multi attention head, Feed forward
# Model - Overall model build
# Also include skip connections, kai ming init, low weight for last layer

import numpy as np

class Tensor:
    def __init__(self, data, symbol=None, requires_grad=True): # symbol for debug
        self.data = np.array(data, dtype=np.float64)
        self.symbol = symbol
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.requires_grad = requires_grad
        self.parents = ()
        self._backward = lambda: None

    def backward(self):
        # Build a topo and recursively backward through parents
        visited = set()
        topo = list()
        def build_topo(ten: Tensor):
            # print(ten.symbol, [par.symbol for par in ten.parents])
            if ten not in visited:
                visited.add(ten)
                for parent in ten.parents:
                    build_topo(parent)
                topo.append(ten)
        build_topo(self)
        for ten in reversed(topo):
            ten._backward()
            
    def __hash__(self):
        return id(self)

    @property
    def shape(self):
        return self.data.shape

    # cast to array when array operation is called on tensor
    def __array__(self, dtype=None):
        return self.data.astype(dtype) if dtype else self.data



    #### Tensor operations, numpy wrapped

    def __matmul__(self, other):
        other = as_tensor(other, requires_grad=False)
    
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
    
        out.parents = (self, other)
    
        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
    
        out._backward = _backward
        return out

    def __add__(self, other):
        other = as_tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        out.parents = (self, other)
    
        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                grad = out.grad
                # sum over broadcasted dims
                while grad.ndim > other.data.ndim:
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(other.data.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad += grad
    
        out._backward = _backward
        return out
    
    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad)
        out.parents = (self, )
        
        def _backward():
            if self.requires_grad:
                self.grad -= out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        """
        Called when self is a tensor, other may be a tensor
        """
        return self + (-other)

    def __rsub__(self, other):
        """
        Called when other is a tensor, but self is not
        """
        return Tensor(self, requires_grad=False) - other 
    
    def __mul__(self, other):
        other = as_tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        out.parents = (self, other)

        # def _backward():
        #     # must check if broadcast occured
        #     # if same dim, then


        
    

    def cross_entropy(self, y, axis=-1):
        # logits: (B, C)
        B, C = self.data.shape
    
        # stable softmax
        x = self.data
        x = x - x.max(axis=1, keepdims=True)
        exp_x = np.exp(x)
        probs = exp_x / exp_x.sum(axis=-1, keepdims=True)
    
        # loss value
        loss_value = -np.log(probs[np.arange(B), y]).mean()
        loss = Tensor(loss_value, symbol="loss")
    
        loss.parents = (self,)
    
        def _backward():
            grad = probs.copy()
            grad[np.arange(B), y] -= 1
            grad /= B
    
            self.grad += grad
    
        loss._backward = _backward
        loss.grad = np.array(1.0)
        return loss

    def debug(self, name=""):
        print(f"\n--- Tensor Debug {name} ---")
        print("id:", id(self))
        print("data:", self.data)
        print("shape:", self.data.shape)
        print("grad:", self.grad)
        print("requires_grad:", self.requires_grad)
        print("parents:", self.parents)
        print("num_parents:", None if self.parents is None else len(self.parents))
        print("backward fn:", self._backward)


def as_tensor(data, requires_grad=True):
    if isinstance(data, Tensor):
        return data
    return Tensor(data, requires_grad=requires_grad)
