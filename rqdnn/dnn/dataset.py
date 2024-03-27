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