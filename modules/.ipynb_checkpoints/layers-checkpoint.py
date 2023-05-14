import numpy as np
from typing import List
from .base import Module


class Linear(Module):
    """
    Applies linear (affine) transformation of data: y = x W^T + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        :param in_features: input vector features
        :param out_features: output vector features
        :param bias: whether to use additive bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(-1, 1, (out_features, in_features)) / np.sqrt(in_features)
        self.bias = np.random.uniform(-1, 1, out_features) / np.sqrt(in_features) if bias else None

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, in_features)
        :return: array of shape (batch_size, out_features)
        """
        if self.bias is not None:
            res = input @ self.weight.T + self.bias
        else:
            res = input @ self.weight.T
        return res

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        :return: array of shape (batch_size, in_features)
        """
        res = grad_output @ self.weight
        return res

    def update_grad_parameters(self, input: np.array, grad_output: np.array):
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        """
        self.grad_weight += grad_output.T @ input
        if self.bias is not None:
            self.grad_bias += np.sum(grad_output, axis=0)

    def zero_grad(self):
        self.grad_weight.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.array]:
        if self.bias is not None:
            return [self.weight, self.bias]

        return [self.weight]

    def parameters_grad(self) -> List[np.array]:
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]

        return [self.grad_weight]

    def __repr__(self) -> str:
        out_features, in_features = self.weight.shape
        return f'Linear(in_features={in_features}, out_features={out_features}, ' \
               f'bias={not self.bias is None})'


class BatchNormalization(Module):
    """
    Applies batch normalization transformation
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        """
        :param num_features:
        :param eps:
        :param momentum:
        :param affine: whether to use trainable affine parameters
        """
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.weight = np.ones(num_features) if affine else None
        self.bias = np.zeros(num_features) if affine else None

        self.grad_weight = np.zeros_like(self.weight) if affine else None
        self.grad_bias = np.zeros_like(self.bias) if affine else None

        # store this values during forward path and re-use during backward pass
        self.mean = None
        self.input_mean = None  # input - mean
        self.var = None
        self.sqrt_var = None
        self.inv_sqrt_var = None
        self.norm_input = None

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """ 
        batch_size = input.shape[0]
        if self.training:
            self.mean = np.mean(input, axis=0)
            self.var = np.mean((input - self.mean) ** 2, axis=0)
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * batch_size / (batch_size - 1) * self.var
            
        else:
            self.mean = self.running_mean.copy()
            self.var = self.running_var.copy()

        self.var += self.eps
        self.sqrt_var = np.sqrt(self.var)
        self.inv_sqrt_var = 1 / self.sqrt_var
        self.input_mean = input - self.mean
        self.norm_input = self.input_mean * self.inv_sqrt_var
        if self.affine:
            return self.weight * self.norm_input + self.bias
        return self.norm_input

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        batch_size = input.shape[0]
        if self.affine:
            standard_grad = grad_output * self.weight
        else:
            standard_grad = grad_output
        if self.training:

            var_grad = np.sum(standard_grad * self.input_mean * -0.5 * self.var ** (-3/2), axis=0)

            aux_x_minus_mean = 2 * self.input_mean / batch_size

            mean_grad = np.sum(standard_grad * -self.inv_sqrt_var, axis=0) + var_grad * np.sum(-aux_x_minus_mean, axis=0)

            res = standard_grad * self.inv_sqrt_var + var_grad * aux_x_minus_mean + mean_grad / batch_size
        else:

            res = standard_grad / np.sqrt(self.running_var + self.eps)
            
        return res

    def update_grad_parameters(self, input: np.array, grad_output: np.array):
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        """
        if self.affine:
            self.grad_weight += np.sum(self.norm_input * grad_output, axis=0)
            self.grad_bias += np.sum(grad_output, axis=0)

    def zero_grad(self):
        if self.affine:
            self.grad_weight.fill(0)
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.array]:
        return [self.weight, self.bias] if self.affine else []

    def parameters_grad(self) -> List[np.array]:
        return [self.grad_weight, self.grad_bias] if self.affine else []

    def __repr__(self) -> str:
        return f'BatchNormalization(num_features={len(self.running_mean)}, ' \
               f'eps={self.eps}, momentum={self.momentum}, affine={self.affine})'


class Dropout(Module):
    """
    Applies dropout transformation
    """
    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.p, input.shape)
            y = 1 / (1 - self.p) * self.mask * input
        else: 
            self.mask = np.ones(input.shape)
            y = input
        return y

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        if self.training:
            res = 1 / (1 - self.p) * self.mask * grad_output
        else:
            res = grad_output
        return res

    def __repr__(self) -> str:
        return f'Dropout(p={self.p})'


class Sequential(Module):
    """
    Container for consecutive application of modules
    """
    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size matching the input size of the first layer
        :return: array of size matching the output size of the last layer
        """
        x = input
        for i in self.modules:
            x = i.compute_output(x)
        return x

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size matching the input size of the first layer
        :param grad_output: array of size matching the output size of the last layer
        :return: array of size matching the input size of the first layer
        """
        num_layers = len(self.modules)
        x = input
        y = grad_output
        results = [x]
        for i in self.modules[:-1]:
            x = i.compute_output(x)
            results.append(x)
        for idx, i in enumerate(self.modules[::-1]):
            x = results[num_layers - idx - 1]
            res = i.compute_grad_input(x, y)
            if not i.__class__.__name__ in ['Sigmoid', 'ReLU', 'Softmax', 'LogSoftmax']: 
                i.update_grad_parameters(x, y)
            y = res
            
            
        return y

    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def parameters(self) -> List[np.array]:
        return [parameter for module in self.modules for parameter in module.parameters()]

    def parameters_grad(self) -> List[np.array]:
        return [grad for module in self.modules for grad in module.parameters_grad()]

    def __repr__(self) -> str:
        repr_str = 'Sequential(\n'
        for module in self.modules:
            repr_str += ' ' * 4 + repr(module) + '\n'
        repr_str += ')'
        return repr_str
