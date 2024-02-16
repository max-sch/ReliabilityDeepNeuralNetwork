import keras
import numpy as np

class Dataset:
    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self):
        if self.idx < self.size():
            x,y = self.X[self.idx], self.Y[self.idx]
            self.idx += 1
            return (x,y)
        else:
            raise StopIteration

    def size(self):
        raise NotImplementedError
    
class MNISTDataset(Dataset):
    def __init__(self, X, Y) -> None:
        super().__init__()
        self.X = X
        self.Y = Y
        self.num_examples = len(X)

    def size(self):
        return self.num_examples

    def create_train():
        (x_train, y_train), _ = keras.datasets.mnist.load_data()
        return MNISTDataset(x_train, y_train)
    
    def create_test(train_size=1000):
        _, (x_test, y_test) = keras.datasets.mnist.load_data()
        #return MNISTDataset(x_test, y_test)
        idx = np.array([1] * train_size + [0] * (len(x_test) - train_size)) > 0
        np.random.shuffle(idx)
        X, Y = x_test[idx,:], y_test[idx]
        return MNISTDataset(X, Y)
    
    def create_cal(cal_size=1000):
        _, (x_test, y_test) = keras.datasets.mnist.load_data()

        idx = np.array([1] * cal_size + [0] * (len(x_test) - cal_size)) > 0
        np.random.shuffle(idx)
        X, Y = x_test[idx,:], y_test[idx]
        return MNISTDataset(X, Y)

