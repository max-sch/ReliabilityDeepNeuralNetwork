from dnn.dataset import Dataset
from dnn.model import Model
from latentspace.clustering import GaussianClusterAnalyzer, estimate_init_means
from reliability.analyzer import ConformalPredictionBasedReliabilityAnalyzer
from evaluation.metrics import AverageReliabilityScores, PearsonCorrelation
from latentspace.partition_map import DecisionTreePartitioning, KnnPartitioning
from evaluation.base import Evaluation
from commons.ops import determine_deviation_softmax

import numpy as np
import keras
from keras import layers


class MNISTEvaluation(Evaluation):
    def evaluate(self):
        models = self._load_models()
        gaussian_cal_set = MNISTDataset.create_first()
        #gaussian_cal_set = MNISTDataset.create_randomly(1000)
        evaluation_set = MNISTDataset.create_second()
        #evaluation_set = MNISTDataset.create_randomly(100)
        # The model will be updated through the evaluation; thus, its temporarily set to None
        rel_analyzer = ConformalPredictionBasedReliabilityAnalyzer(model=None,
                                                                   calibration_set=MNISTDataset.create_randomly(),
                                                                   tuning_set=MNISTDataset.create_randomly())
        metrics = [AverageReliabilityScores(), 
                   PearsonCorrelation(determine_deviation=lambda softmax, true_labels: determine_deviation_softmax(softmax, 
                                                                                                                   true_labels, 
                                                                                                                   rel_analyzer.class_to_idx_mapper))
        ]
        partitioning_algs = [DecisionTreePartitioning(), KnnPartitioning()]

        super().evaluate(models=models,
                         evaluation_set=evaluation_set,
                         gaussian_cal_set=gaussian_cal_set,
                         rel_analyzer=rel_analyzer,
                         partition_algs=partitioning_algs,
                         metrics=metrics,
                         include_softmax=True)

    def train_models(self):
        train_data = MNISTDataset.create_less_train()
        test_data = MNISTDataset.create_test()

        model = MNISTTestModel3(model_file=None)
        model.train_and_save_model(train_data=train_data, test_data=test_data)

    def estimate_gaussian(self, features, predictions):
        means_init = estimate_init_means(features, predictions, num_labels=10)
        cluster_analyzer = GaussianClusterAnalyzer(means_init)
        cluster_analyzer.estimate(features)
        return cluster_analyzer

    def _load_models(self):
        return [MNISTTestModel3()]
    
    def _load_std_metrics(self):
        pears_corr = PearsonCorrelation(determine_deviation=lambda softmax, true_labels: determine_deviation_softmax(softmax, 
                                                                                                                     true_labels, 
                                                                                                                     class_to_idx_mapper=lambda x: x))
        std_metrics = super()._load_std_metrics()
        std_metrics.append(pears_corr)
        return std_metrics
    
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
    
class MNISTTestModel3(Model):
    def __init__(self, model_file="MNISTTestModel3.keras") -> None:
        self.name = "MNISTTestModel3"
        self.num_classes = 10
        self.input_shape = (28, 28, 1)
        if not model_file==None:
            self.model = self.load_from(model_file)
            
            self.feature_extractor = self.load_from(model_file)
            self.feature_extractor.pop()
            self.feature_extractor.pop()
            self.feature_extractor.summary()
        else:
            self.model = keras.Sequential(
                [
                    keras.Input(shape=self.input_shape),
                    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Dropout(0.25),
                    layers.Flatten(),
                    layers.Dense(128, activation='relu'),
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

        self.model.save("MNISTTestModel3.keras")

    def load_from(self, model_file):
        return keras.saving.load_model(model_file)

    def get_confidences(self, x) -> dict:
        x = self._prepare_input(x)
        return {class_idx: float(probability) for class_idx, probability in enumerate(self.model(x)[0])}

    def predict(self, x):
        x = self._prepare_input(x)
        softmax_predictions = self.model(x)
        return int(np.argmax(softmax_predictions, axis=1))
    
    def predict_all(self, X):
        softmax_predictions = self.softmax(X)
        return np.argmax(softmax_predictions, axis=1)
    
    def softmax(self, X):
        X_prep = self._prepare_x_data(X)
        return self.model(X_prep)
    
    def get_confidences_for_feature(self, feature) -> dict:
        softmax = self.model.layers[-1]
        softmax_predictions = softmax(feature)
        test = {class_idx: float(probability) for class_idx, probability in enumerate(softmax_predictions[0])}
        return test
    
    def confidence(self, x, y): 
        return self.get_confidences(x)[y]

    def project(self, x):
        x = self._prepare_input(x)
        return [float(element) for element in self.feature_extractor(x)[0]]
    
    def project_all(self, X):
        X_prep = self._prepare_x_data(X)
        return self.feature_extractor(X_prep)
    
    def _prepare_input(self, X):
        return self._prepare_x_data(X).reshape(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
    
    def _prepare_x_data(self, X):
        # Scale images to the [0, 1] range
        X_prep = X.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        return np.expand_dims(X_prep, -1)

    def _prepare_y_data(self, Y):
        return keras.utils.to_categorical(Y, self.num_classes)