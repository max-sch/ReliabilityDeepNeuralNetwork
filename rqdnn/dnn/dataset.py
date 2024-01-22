import keras

class Dataset:
    def __iter__(self):
        self.idx = 1
        return self
    
    def __next__(self):
        x,y = self.X[self.idx], self.Y[self.idx]
        self.idx += 1
        return (x,y)

    def size(self):
        raise NotImplementedError
    
class MNISTDataset(Dataset):
    def __init__(self, X, Y) -> None:
        super().__init__()
        self.X = X
        self.Y = Y

    def create_train():
        (x_train, y_train), _ = keras.datasets.mnist.load_data()
        return MNISTDataset(x_train, y_train)
    
    def create_test():
        _, (x_test, y_test) = keras.datasets.mnist.load_data()
        return MNISTDataset(x_test, y_test)
