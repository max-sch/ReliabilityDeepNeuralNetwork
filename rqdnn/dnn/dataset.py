import keras
import numpy as np

class Dataset:
    def __init__(self, X, Y) -> None:
        self.X = X
        self.Y = Y
        self.num_examples = len(X)

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
        return self.num_examples
    
class MNISTDataset(Dataset):
    def __init__(self, X, Y) -> None:
        super().__init__(X, Y)

    def create_train():
        (x_train, y_train), _ = keras.datasets.mnist.load_data()
        return MNISTDataset(x_train, y_train)
    
    def create_less_train():
        (x_train, y_train), _ = keras.datasets.mnist.load_data()

        idx = np.array([1] * 100 + [0] * (len(x_train) - 100)) > 0
        np.random.shuffle(idx)
        X, Y = x_train[idx,:], y_train[idx]
        return MNISTDataset(X, Y)
    
    def create_test():
        _, (x_test, y_test) = keras.datasets.mnist.load_data()
        return MNISTDataset(x_test, y_test)
        
    def create_randomly(size=1000):
        _, (x_test, y_test) = keras.datasets.mnist.load_data()

        idx = np.array([1] * size + [0] * (len(x_test) - size)) > 0
        np.random.shuffle(idx)
        X, Y = x_test[idx,:], y_test[idx]
        return MNISTDataset(X, Y)
    
    def create_first():
        _, (x_test, y_test) = keras.datasets.mnist.load_data()
        
        idx = range(5000)
        X, Y = x_test[idx,:], y_test[idx]
        return MNISTDataset(X, Y)
    
    def create_second():
        _, (x_test, y_test) = keras.datasets.mnist.load_data()
        
        idx = range(5000, 10000)
        X, Y = x_test[idx,:], y_test[idx]
        return MNISTDataset(X, Y)