import random
from micrograd.engine import Value
import numpy as np

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
        dim = x.shape[0]
        if len(x.shape)==2:
            dim = x.shape[1]
        assert dim == len(self.w), f"input shape to neuron mismatch, expected {len(self.w)} but got {dim}"
        act = np.dot(x, self.w) + self.b
        tanh_output = np.vectorize(lambda x: x.tanh())
        return tanh_output(act) if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = np.array([n(x) for n in self.neurons])
        out = np.transpose(out)
        return out[0] if len(out) == 1 else np.array(out)

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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, nonlin=True):

        self.weights = np.empty((out_channels, in_channels, kernel_size, kernel_size), dtype=Value)
        for i in range(out_channels):
            single_kernel = [Value(random.uniform(-1,1)) for _ in range(in_channels*(kernel_size**2))]
            self.weights[i] = np.reshape(np.array(single_kernel),(in_channels, kernel_size, kernel_size))

        self.biases = np.array([Value(random.uniform(-1,1)) for _ in range(out_channels)])

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.nonlin = nonlin

    """
        input shape (N, C, H, W) or (C, H, W) where:
            N: batch size
            C: number of channels
            H: height
            W: width
    """
    def __call__(self, input):

        assert len(input.shape) == 3 or len(input.shape) == 4, 'The input must be either 4D (batched) or 3D (unbatched)'

        if len(input.shape) == 3:
            input = np.expand_dims(input, axis=0)

        padded = self._pad(input)
        
        batch_size, in_channels, height, width = padded.shape
        
        def calculate_dim(input_dim):
            return np.floor((input_dim - self.kernel_size) / self.stride).astype(int) + 1

        output_dims = (batch_size, self.out_channels, calculate_dim(height), calculate_dim(width))
        output = np.empty(output_dims, dtype=Value)

        for h in range(output.shape[2]):
            for w in range(output.shape[3]):
                receptive_field = padded[:, :, h : h + self.kernel_size, w : w + self.kernel_size]
                temp_weights = np.expand_dims(self.weights, axis=0)
                receptive_field = np.expand_dims(receptive_field, axis=1)
                output[:, :, h, w] = np.sum(receptive_field * temp_weights, axis=(2, 3, 4)) + self.biases

        if self.nonlin:
            # relu_output = np.vectorize(lambda x: x.relu())
            # output = relu_output(output)
            tanh_output = np.vectorize(lambda x: x.tanh())
            output = tanh_output(output)
        return output

    def _pad(self, x):
        n, c, h, w = x.shape

        horizontal_zeros = np.zeros((n, c, self.padding, w), dtype=Value)
        x = np.append(horizontal_zeros, x, axis=2)
        x = np.append(x, horizontal_zeros, axis=2)
        vertical_zeros = np.zeros((n, c, h + self.padding*2, self.padding), dtype=Value)
        x = np.append(vertical_zeros, x, axis=3)
        x = np.append(x, vertical_zeros, axis=3)

        return x

    def parameters(self):
        w_1d = self.weights.flatten()
        params = np.append(w_1d, self.biases)
        return params
    
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'} Convolution with shape=({self.weights.shape}), padding=({self.padding}), stride=({self.stride})"

class CNN(Module):

    def __init__(self, conv_layers, mlp, is_softmax=True):
        self.conv_layers = conv_layers
        self.mlp = mlp
        self.is_softmax = is_softmax

    def __call__(self, x):
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], -1)
        else:
            x = x.flatten()
        x = self.mlp(x)
        if self.is_softmax:
            x = self._softmax(x)
        return x

    def _softmax(self, x):
        denominator = np.sum(np.exp(x), axis=1, keepdims=True) if len(x.shape) == 2 else np.sum(np.exp(x))
        return np.exp(x)/denominator

    def parameters(self):
        return np.append([p for conv in self.conv_layers for p in conv.parameters()], self.mlp.parameters())

    def __repr__(self):
        return f"{'Softmax' if self.is_softmax else ''} CNN of convs [{', '.join(str(conv) for conv in self.conv_layers)}] and mlp [{str(self.mlp)}]"
