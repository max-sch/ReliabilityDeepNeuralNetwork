class Model:
    def __init__(self, name, model_file) -> None:
        self.name = name
        self.model = self.load_from(model_file)

    def load_from(self, model_file):
        '''Loads the dnn model from specified file'''
        raise NotImplementedError

    def get_confidences(self, x) -> dict:
        '''Returns the prediction confidence associated with input x'''
        raise NotImplementedError
    
    def softmax(self, X):
        '''Returns the soft max results for input batch X. Must only be implemented, for classifiers.'''
        raise NotImplementedError

    def get_confidences_for_feature(self, feature) -> dict:
        '''Returns the prediction confidence associated with the given feature'''
        raise NotImplementedError

    def predict(self, x):
        '''Predicts output y for input x'''
        raise NotImplementedError
    
    def predict_all(self, X):
        '''Predicts outputs Y for input batch X'''
        raise NotImplementedError
    
    def confidence(self, x, y):
        '''Returns the prediction confidence associated with output y given input x'''
        raise NotImplementedError

    def project(self, x):
        '''Projects input x to the feature space'''
        raise NotImplementedError
    
    def project_all(self, X):
        '''Projects a batch of inputs X to the feature space'''
        raise NotImplementedError
    
    def get_output_shape(self):
        return self.model.layers[-1].output_shape[1]
    
