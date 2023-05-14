import numpy as np
import scipy
from .base import Module


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.maximum(input, np.zeros(input.shape))

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return np.where(input>0,1,0) * grad_output


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return 1 / (1 + np.exp(- input))

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return np.exp(input) / (np.exp(input) + 1) ** 2 * grad_output


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return scipy.special.softmax(input, axis=1)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        s = self.compute_output(input)
        num_examples = input.shape[1]
        arr = np.empty((0, num_examples))
        for i in range(input.shape[0]):
            Sz = s[i]
            D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
            arr = np.append(arr, (D @ grad_output[i]).reshape((1, num_examples)), axis=0)
        return arr
    
    
class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return scipy.special.log_softmax(input, axis=1)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        s = scipy.special.softmax(input, axis=1)
        num_examples = input.shape[1]
        arr = np.empty((0, num_examples))
        for i in range(input.shape[0]):
            Sz = s[i]
            D = np.identity(num_examples) - np.repeat(Sz,num_examples).reshape((num_examples,num_examples))
            arr = np.append(arr, (D @ grad_output[i]).reshape((1, num_examples)), axis=0)
        return arr
