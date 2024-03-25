from dnn.dataset import MNISTDataset
from dnn.model import MNISTTestModel, MNISTTestModel2, MNISTTestModel3
from latentspace.clustering import GaussianClusterAnalyzer, estimate_init_means
from reliability.analyzer import ConformalPredictionBasedReliabilityAnalyzer
from evaluation.metrics import AverageReliabilityScores
from latentspace.partition_map import DecisionTreePartitioning, KnnPartitioning
from evaluation.base import Evaluation

class MNISTEvaluation(Evaluation):
    def evaluate(self):
        models = self._load_models()
        gaussian_cal_set = MNISTDataset.create_first()
        #gaussian_cal_set = MNISTDataset.create_randomly(1000)
        evaluation_set = MNISTDataset.create_second()
        #evaluation_set = MNISTDataset.create_randomly(1000)
        # The model will be updated through the evaluation; thus, its temporarily set to None
        rel_analyzer = ConformalPredictionBasedReliabilityAnalyzer(model=None,
                                                                   calibration_set=MNISTDataset.create_randomly(),
                                                                   tuning_set=MNISTDataset.create_randomly())
        metrics = [AverageReliabilityScores()]
        partitioning_algs = [DecisionTreePartitioning(), KnnPartitioning()]

        super().evaluate(models=models,
                         evaluation_set=evaluation_set,
                         gaussian_cal_set=gaussian_cal_set,
                         rel_analyzer=rel_analyzer,
                         partition_algs=partitioning_algs,
                         metrics=metrics)

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