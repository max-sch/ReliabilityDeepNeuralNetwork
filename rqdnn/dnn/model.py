class Model:
    def __init__(self, name, model_file) -> None:
        self.name = name
        self.model = self.load_from(model_file)

    def load_from(self, model_file):
        '''Loads the dnn model from specified file'''
        raise NotImplementedError
    
    def softmax(self, X):
        '''Returns the soft max results for input batch X. Must only be implemented, for classifiers.'''
        raise NotImplementedError

    def softmax_for_features(self, features) -> dict:
        '''Returns the prediction confidence or softmax associated with the given features'''
        raise NotImplementedError

    def predict(self, X):
        '''Predicts outputs Y for input batch X'''
        raise NotImplementedError
    
    def project(self, X):
        '''Projects a batch of inputs X to the feature space'''
        raise NotImplementedError
    
    def get_output_shape(self):
        return self.model.layers[-1].output_shape[1]
    
