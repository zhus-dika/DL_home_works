import numpy as np
from .base import Criterion
from .activations import LogSoftmax, Softmax

class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.array, target: np.array) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        res = np.sum(np.sum((input - target) ** 2, axis = 0), axis = 0)
        return res / (target.shape[0] * target.shape[1])

    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        return 2 * (input - target) / (target.shape[0] * target.shape[1])


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()
        self.softmax = Softmax()

    def compute_output(self, input: np.array, target: np.array) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        batch_size = input.shape[0]
        num_classes = input.shape[1]
        target_one_hot = np.eye(input.shape[0], input.shape[1])[target]
        res = - np.sum(np.sum(self.log_softmax.compute_output(input) * target_one_hot, axis=0), axis=0) / batch_size
        return res

    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        target_one_hot = np.eye(input.shape[0], input.shape[1])[target]
        return (self.softmax.compute_output(input) - target_one_hot) / input.shape[0]
