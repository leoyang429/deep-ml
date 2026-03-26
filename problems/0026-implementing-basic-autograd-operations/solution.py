class Value:
    def __init__(self, data, _children=(), _op='', _backward=lambda x: None):
        self.data = data
        self.grad = 0
        self._backward = _backward
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={int(self.grad)})"

    def __add__(self, other):
        # Implement addition here
        def _backward(_grad):
            self.grad += 1 * _grad
            other.grad += 1 * _grad
        return Value(self.data + other.data, (self, other), '+', _backward)

    def __mul__(self, other):
        # Implement multiplication here
        def _backward(_grad):
            self.grad += other.data * _grad
            other.grad += self.data * _grad
        return Value(self.data * other.data, (self, other), '*', _backward)

    def relu(self):
        # Implement ReLU here
        def _backward(_grad):
            self.grad += 1 * _grad if self.data > 0 else 0
        return Value(self.data if self.data > 0 else 0, (self, ), 'ReLU', _backward)

    def backward(self, _grad=None):
        # Implement backward pass here
        if _grad is None:
            self.grad = 1.0
        self._backward(self.grad)
        for child in self._prev:
            child.backward(self.grad)