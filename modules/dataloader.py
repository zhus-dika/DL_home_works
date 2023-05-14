import numpy as np
class DataLoader(object):
    """
    Tool for shuffling data and forming mini-batches
    """
    def __init__(self, X, y, batch_size=1, shuffle=False):
        """
        :param X: dataset features
        :param y: dataset targets
        :param batch_size: size of mini-batch to form
        :param shuffle: whether to shuffle dataset
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0  # use in __next__, reset in __iter__

    def __len__(self) -> int:
        """
        :return: number of batches per epoch
        """
        return int(self.X.shape[0] / self.batch_size) + (self.X.shape[0] % self.batch_size > 0)

    def num_samples(self) -> int:
        """
        :return: number of data samples
        """
        return self.X.shape[0]

    def __iter__(self):
        """
        Shuffle data samples if required
        :return: self
        """
        if self.shuffle:
            shuffled_indices = np.random.permutation(self.num_samples()) #return a permutation of the indices
            self.X = self.X[shuffled_indices]
            self.y = self.y[shuffled_indices]
        self.batch_id = 0
        return self

    def __next__(self):
        """
        Form and return next data batch
        :return: (x_batch, y_batch)
        """
        self.batch_id += 1
        lenb = self.__len__()
        if self.batch_id <= lenb:
            nsamples = self.num_samples()
            index = (self.batch_id - 1) * self.batch_size
            if index + self.batch_size >= nsamples:
                return (self.X[index:nsamples], self.y[index:nsamples])
            else:
                return (self.X[index:index + self.batch_size], self.y[index:index + self.batch_size])
        raise StopIteration
