import random
from engine import Value
import numpy as np
import itertools

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

class Convolution(Module):

    def __init__(self, size, n_channels, padding=0, stride=1, nonlin=True):
        self.w = np.array([[[Value(random.uniform(-1,1)) for _ in range(n_channels)] for _ in range(size)] for _ in range(size)])
        self.b = Value(0)
        self.n_channels = n_channels
        self.size = size
        self.padding = padding
        self.stride = stride
        self.nonlin = nonlin

    def __call__(self, x):
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        assert x.shape[2] == self.n_channels, 'shape mismatch, the number of channels of input isn\'t equal to the conv channels'
        padded = self._pad(x)
        output_dims = (np.floor((padded.shape[0] - self.size + 1.0) / self.stride).astype(int),
                        np.floor((padded.shape[1] - self.size + 1.0) / self.stride).astype(int))
        output = np.zeros(output_dims, dtype=Value)
        for i in range(output_dims[0]):
            for j in range(output_dims[1]):
                starting_i = i * self.stride
                starting_j = j * self.stride
                sub_image = padded[starting_i : starting_i + self.size, starting_j : starting_j + self.size, :]
                output[i][j] = np.sum(np.multiply(sub_image, self.w)) + self.b
        return output

    def _pad(self, x):
        h, w, c = x.shape
        horizontal_zeros = np.zeros((self.padding, w, c),dtype=Value)
        x = np.append(horizontal_zeros, x, axis=0)
        x = np.append(x, horizontal_zeros, axis=0)
        vertical_zeros = np.zeros((h + self.padding*2, self.padding, c),dtype=Value)
        x = np.append(vertical_zeros, x, axis=1)
        x = np.append(vertical_zeros, x, axis=1)
        return x

    def parameters(self):
        w_1d = self.w.flatten()
        params = np.append(w_1d,self.b)
        return params

np.random.seed(1337)
random.seed(1337)
conv = Convolution(1, 1, padding=0, stride=1)
# x = np.array([[[1,2,3],[1,2,3]],
#               [[1,2,3],[1,2,3]],
#               [[1,2,3],[1,2,3]],
#               [[1,2,3],[1,2,3]]
#               ])

# x = np.array([[[1,1],[1,2]],[[1,1],[1,2]]])
# x = np.array[[1,1,1],[1,2,3]])
x = np.array([[1,2],[3,4]])
# x = np.array([[3]])
conv(x)
print(conv.parameters())

