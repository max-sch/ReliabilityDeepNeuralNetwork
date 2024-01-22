import numpy as np
import keras
from keras import layers

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
    
    def confidence(self, x, y):
        '''Returns the prediction confidence associated with output y given input x'''
        raise NotImplementedError

    #def prediction_with_confidences(self, x):
    #    '''Predicts output y for input x and returns confidences'''
    #    confidences = self.get_confidences(x)
    #    return (max(confidences, key=confidences.get), confidences)

    def project(self, x):
        '''Projects input x to the feature space'''
        raise NotImplementedError
    
class MNISTTestModel(Model):
    def __init__(self, model_file="MNISTTestModel.keras") -> None:
        self.num_classes = 10
        self.input_shape = (28, 28, 1)
        if not model_file==None:
            self.model = self.load_from(model_file)
            
            self.feature_extractor = self.load_from(model_file)
            self.feature_extractor.summary()
            self.feature_extractor.pop()
            self.feature_extractor.pop()
            self.feature_extractor.summary()
        else:
            self.model = keras.Sequential(
                [
                    keras.Input(shape=self.input_shape),
                    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Flatten(),
                    layers.Dropout(0.5),
                    layers.Dense(self.num_classes, activation="softmax"),
                ]
            )
            self.model.summary()

            self.feature_extractor = None

    def train_and_save_model(self, train_data, test_data):
        batch_size = 128
        epochs = 15

        self.model.compile(loss="categorical_crossentropy", 
                           optimizer="adam", 
                           metrics=["accuracy"])

        x_train = self._prepare_x_data(train_data.X)
        y_train = self._prepare_y_data(train_data.Y)

        self.model.fit(x_train, 
                       y_train, 
                       batch_size=batch_size, 
                       epochs=epochs, 
                       validation_split=0.1)
        
        x_test = self._prepare_x_data(test_data.X)
        y_test = self._prepare_y_data(test_data.Y)
        
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

        self.model.save("MNISTTestModel.keras")

    def load_from(self, model_file):
        return keras.saving.load_model(model_file)

    def get_confidences(self, x) -> dict:
        x = self._prepare_input(x)
        return {class_idx: float(probability) for class_idx, probability in enumerate(self.model(x)[0])}

    def predict(self, x):
        x = self._prepare_input(x)
        soft_max_predictions = self.model(x)
        return int(np.argmax(soft_max_predictions, axis=1))
    
    def confidence(self, x, y): 
        return self.get_confidences(x)[y]

    def project(self, x):
        x = self._prepare_input(x)
        return [float(element) for element in self.feature_extractor(x)[0]]
    
    def _prepare_input(self, X):
        return self._prepare_x_data(X).reshape(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
    
    def _prepare_x_data(self, X):
        # Scale images to the [0, 1] range
        X_prep = X.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        return np.expand_dims(X_prep, -1)

    def _prepare_y_data(self, Y):
        return keras.utils.to_categorical(Y, self.num_classes)
    
