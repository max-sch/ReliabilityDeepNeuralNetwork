class Model:
    def __init__(self, model_file) -> None:
        self.model = self.load_from(model_file)

    def load_from(self, model_file):
        '''Loads the dnn model from specified file'''
        raise NotImplementedError

    def get_confidences(self, x) -> dict:
        '''Returns the prediction confidence associated with input x'''
        raise NotImplementedError

    def predict(self, x):
        '''Predicts output y for input x'''
        raise NotImplementedError

    def prediction_with_confidences(self, x):
        '''Predicts output y for input x and returns confidences'''
        confidences = self.get_confidences(x)
        return (max(confidences, key=confidences.get), confidences)

    def project(self, x):
        '''Projects input x to the feature space'''
        raise NotImplementedError