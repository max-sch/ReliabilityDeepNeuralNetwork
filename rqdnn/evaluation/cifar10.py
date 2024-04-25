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

idx_to_class_map = {"airplane":0, "automobile":1, "bird":2, "cat":3, "deer":4, "dog":5, "frog":6, "horse":7, "ship":8, "truck":9,}

class_to_idx_mapper = lambda x:x

def determine_softmax_pos_fun(softmax, true_labels):
    return determine_deviation_softmax(softmax, true_labels, class_to_idx_mapper) 

class CIFAR10Evaluation(Evaluation):
    def __init__(self) -> None:
        self.mnist_provider = CFIR10DatasetProvider()

    def evaluate(self):
        models = self._load_models()
        gaussian_cal_set = self.mnist_provider.create_gaussian_cal()
        evaluation_set = self.mnist_provider.create_evaluation()

        metrics = [AverageReliabilityScores()]
        partitioning_algs = [DecisionTreePartitioning(), KnnPartitioning()]

        super().evaluate(models=models,
                         evaluation_set=evaluation_set,
                         gaussian_cal_set=gaussian_cal_set,
                         partition_algs=partitioning_algs,
                         metrics=metrics,
                         include_softmax=True)

    def train_models(self):
        train_data = self.mnist_provider.create_full_train()
        test_data = self.mnist_provider.create_evaluation()

        model = CFIR10Model(model_level=ModelLevel.BEST, load_model=False)
        model.train_and_save_model(train_data, test_data)

        train_data = self.mnist_provider.create_half_train()

        model = CFIR10Model(model_level=ModelLevel.AVG, load_model=False)
        model.train_and_save_model(train_data, test_data)

        train_data = self.mnist_provider.create_one_percent_train()

        model = CFIR10Model(model_level=ModelLevel.WORST, load_model=False)
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
        #return [CFIR10Model(ModelLevel.BEST), CFIR10Model(ModelLevel.AVG), CFIR10Model(ModelLevel.WORST)]
        return [CFIR10Model(ModelLevel.WORST)]
    
    def _load_std_metrics(self):
        std_metrics = super()._load_std_metrics()
        std_metrics.append(AverageOutputDeviation(determine_deviation=determine_softmax_pos_fun))
        std_metrics.append(SoftmaxPositionToReliabilityCorrelation(determine_deviation=determine_softmax_pos_fun, num_pos=10))
        std_metrics.append(PearsonCorrelation(determine_deviation=determine_softmax_pos_fun))
        return std_metrics
    
class CFIR10DatasetProvider:
    def __init__(self) -> None:
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.cifar10.load_data()

        train_cal_split = random_splits([40000, 10000])
        self.train_idxs = train_cal_split == 0
        self.gaussian_cal_idxs = train_cal_split == 1

        eval_cal_tun_split = random_splits([8000, 1000, 1000])
        self.eval_idxs = eval_cal_tun_split == 0
        self.cal_idxs = eval_cal_tun_split == 1
        self.tun_idxs = eval_cal_tun_split == 2

    def create_full_train(self):
        X, Y = self.x_train[self.train_idxs,:], self.y_train[self.train_idxs][:,0]
        return Dataset(X, Y)
    
    def create_half_train(self):
        x_train, y_train = self.x_train[self.train_idxs,:], self.y_train[self.train_idxs][:,0]
        half_train_split = random_splits([20000, 20000]) == 0
        X, Y = x_train[half_train_split,:], y_train[half_train_split]
        return Dataset(X, Y)
    
    def create_one_percent_train(self):
        x_train, y_train = self.x_train[self.train_idxs,:], self.y_train[self.train_idxs][:,0]
        one_percent_train_split = random_splits([400, 39600]) == 0
        X, Y = x_train[one_percent_train_split,:], y_train[one_percent_train_split]
        return Dataset(X, Y)
    
    def create_gaussian_cal(self):
        X, Y = self.x_train[self.gaussian_cal_idxs,:], self.y_train[self.gaussian_cal_idxs][:,0]
        return Dataset(X, Y)
    
    def create_cal(self):
        X, Y = self.x_test[self.cal_idxs,:], self.y_test[self.cal_idxs][:,0]
        return Dataset(X, Y)
    
    def create_tuning(self):
        X, Y = self.x_test[self.tun_idxs,:], self.y_test[self.tun_idxs][:,0]
        return Dataset(X, Y)

    def create_evaluation(self):
        X, Y = self.x_test[self.eval_idxs,:], self.y_test[self.eval_idxs][:,0]
        return Dataset(X, Y)
    
class CFIR10Model(Model):
    def __init__(self, model_level, load_model=True) -> None:
        self.name = "CFIR10Model_{level}".format(level=model_level.value)
        self.model_file = self.name + ".keras"
        self.num_classes = 10
        self.input_shape = (32, 32, 3)
        if load_model:
            self.model = self.load_from(self.model_file)
            
            self.feature_extractor = self.load_from(self.model_file)
            self.feature_extractor.pop()
            self.feature_extractor.pop()
            self.feature_extractor.summary()
        else:
            self.model = keras.Sequential()
            self.model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=self.input_shape))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.MaxPooling2D(pool_size=(2,2)))
            self.model.add(layers.Dropout(0.3))

            self.model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.MaxPooling2D(pool_size=(2,2)))
            self.model.add(layers.Dropout(0.5))

            self.model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.MaxPooling2D(pool_size=(2,2)))
            self.model.add(layers.Dropout(0.5))

            self.model.add(layers.Flatten())
            self.model.add(layers.Dense(128, activation='relu'))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Dropout(0.5))
            self.model.add(layers.Dense(self.num_classes, activation='softmax'))
            
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
        print("Test accuracy:", score[1])

        self.model.save(self.model_file)

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
        return X_prep
        # Make sure images have shape (28, 28, 1)
        #return np.expand_dims(X_prep, -1)

    def _prepare_y_data(self, Y):
        return keras.utils.to_categorical(Y, self.num_classes)