from dnn.dataset import Dataset
from dnn.model import Model
from latentspace.clustering import GaussianClusterAnalyzer, estimate_init_means
from reliability.analyzer import ConformalPredictionBasedReliabilityAnalyzer
from evaluation.metrics import AverageReliabilityScores, AverageOutputDeviation, SoftmaxPositionToReliabilityCorrelation, PearsonCorrelation
from latentspace.partition_map import DecisionTreePartitioning, KnnPartitioning
from evaluation.base import Evaluation, ModelLevel
from commons.ops import determine_deviation_softmax, random_splits

import numpy as np
import keras
from keras import layers
from keras import metrics

class_to_idx_mapper = lambda x:x

def determine_softmax_pos_fun(softmax, true_labels):
    return determine_deviation_softmax(softmax, true_labels, class_to_idx_mapper) 

class FashionMNISTEvaluation(Evaluation):
    def __init__(self) -> None:
        self.mnist_provider = FashionMNISTDatasetProvider()

    def evaluate(self):
        models = self._load_models()
        gaussian_cal_set = self.mnist_provider.create_gaussian_cal()
        evaluation_set = self.mnist_provider.create_evaluation()

        metrics = [AverageReliabilityScores()]
        partitioning_algs = [DecisionTreePartitioning(), KnnPartitioning()]

        return super().evaluate(models=models,
                                evaluation_set=evaluation_set,
                                gaussian_cal_set=gaussian_cal_set,
                                partition_algs=partitioning_algs,
                                metrics=metrics,
                                include_softmax=True)

    def train_models(self):
        train_data = self.mnist_provider.create_full_train()
        test_data = self.mnist_provider.create_evaluation()

        model = FashionMNISTModel(model_level=ModelLevel.BEST, load_model=False)
        model.train_and_save_model(train_data, test_data)

        train_data = self.mnist_provider.create_half_train()

        model = FashionMNISTModel(model_level=ModelLevel.AVG, load_model=False)
        model.train_and_save_model(train_data, test_data)

        train_data = self.mnist_provider.create_one_percent_train()

        model = FashionMNISTModel(model_level=ModelLevel.WORST, load_model=False)
        model.train_and_save_model(train_data, test_data)

    def estimate_gaussian(self, features, predictions):
        means_init = estimate_init_means(features, predictions, num_labels=10)
        cluster_analyzer = GaussianClusterAnalyzer(means_init)
        cluster_analyzer.estimate(features)
        return cluster_analyzer
    
    def create_rel_analyzer_for(self, model):
        return ConformalPredictionBasedReliabilityAnalyzer(model=model,
                                                           calibration_set=self.mnist_provider.create_cal(),
                                                           tuning_set=self.mnist_provider.create_tuning(),
                                                           class_to_idx_mapper=class_to_idx_mapper)

    def _load_models(self):
        return [FashionMNISTModel(ModelLevel.BEST), FashionMNISTModel(ModelLevel.AVG), FashionMNISTModel(ModelLevel.WORST)]
    
    def load_std_metrics(self):
        std_metrics = super().load_std_metrics()
        std_metrics.append(AverageOutputDeviation(determine_deviation=determine_softmax_pos_fun))
        std_metrics.append(SoftmaxPositionToReliabilityCorrelation(determine_deviation=determine_softmax_pos_fun, num_pos=10))
        std_metrics.append(PearsonCorrelation(determine_deviation=determine_softmax_pos_fun))
        return std_metrics
    
class FashionMNISTDatasetProvider:
    def __init__(self) -> None:
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.fashion_mnist.load_data()

        train_cal_split = random_splits([50000, 10000])
        self.train_idxs = train_cal_split == 0
        self.gaussian_cal_idxs = train_cal_split == 1

        eval_cal_tun_split = random_splits([8000, 1000, 1000])
        self.eval_idxs = eval_cal_tun_split == 0
        self.cal_idxs = eval_cal_tun_split == 1
        self.tun_idxs = eval_cal_tun_split == 2

    def create_full_train(self):
        X, Y = self.x_train[self.train_idxs,:], self.y_train[self.train_idxs]
        return Dataset(X, Y)
    
    def create_half_train(self):
        x_train, y_train = self.x_train[self.train_idxs,:], self.y_train[self.train_idxs]
        half_train_split = random_splits([25000, 25000]) == 0
        X, Y = x_train[half_train_split,:], y_train[half_train_split]
        return Dataset(X, Y)
    
    def create_one_percent_train(self):
        x_train, y_train = self.x_train[self.train_idxs,:], self.y_train[self.train_idxs]
        one_percent_train_split = random_splits([500, 49500]) == 0
        X, Y = x_train[one_percent_train_split,:], y_train[one_percent_train_split]
        return Dataset(X, Y)
    
    def create_gaussian_cal(self):
        X, Y = self.x_train[self.gaussian_cal_idxs,:], self.y_train[self.gaussian_cal_idxs]
        return Dataset(X, Y)
    
    def create_cal(self):
        X, Y = self.x_test[self.cal_idxs,:], self.y_test[self.cal_idxs]
        return Dataset(X, Y)
    
    def create_tuning(self):
        X, Y = self.x_test[self.tun_idxs,:], self.y_test[self.tun_idxs]
        return Dataset(X, Y)

    def create_evaluation(self):
        X, Y = self.x_test[self.eval_idxs,:], self.y_test[self.eval_idxs]
        return Dataset(X, Y)
    
class FashionMNISTModel(Model):
    def __init__(self, model_level, load_model=True) -> None:
        self.name = "FashionMNISTModel_{level}".format(level=model_level.value)
        self.model_file = self.name + ".keras"
        self.num_classes = 10
        self.input_shape = (28, 28, 1)
        if load_model:
            self.model = self.load_from(self.model_file)
            
            self.feature_extractor = self.load_from(self.model_file)
            self.feature_extractor.pop()
            self.feature_extractor.pop()
            self.feature_extractor.summary()
        else:
            self.model = keras.Sequential(
                [
                    keras.Input(shape=self.input_shape),
                    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Dropout(0.25),
                    layers.Flatten(),
                    layers.Dense(256, activation='relu'),
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
                           metrics=[metrics.MeanSquaredError(),
                                    metrics.Accuracy()])

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
        print("MSE:", score[1])
        print("Test accuracy:", score[2])

        self.model.save(self.model_file)

    def load_from(self, model_file):
        return keras.saving.load_model(model_file)
    
    def predict(self, X):
        softmax_predictions = self.softmax(X)
        return np.argmax(softmax_predictions, axis=1)
    
    def softmax(self, X):
        X_prep = self._prepare_x_data(X)
        return self.model(X_prep)
    
    def softmax_for_features(self, features):
        softmax = self.model.layers[-1]
        return softmax(features)
    
    def project(self, X):
        X_prep = self._prepare_x_data(X)
        return self.feature_extractor(X_prep)
    
    def calc_mse(self, dataset):
        self.model.compile(metrics=[metrics.MeanSquaredError()])
        x_test = self._prepare_x_data(dataset.X)
        y_test = self._prepare_y_data(dataset.Y)
        
        return self.model.evaluate(x_test, y_test, verbose=0)[1]
    
    def _prepare_x_data(self, X):
        # Scale images to the [0, 1] range
        X_prep = X.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        return np.expand_dims(X_prep, -1)

    def _prepare_y_data(self, Y):
        return keras.utils.to_categorical(Y, self.num_classes)