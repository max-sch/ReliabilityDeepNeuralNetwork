from latentspace.partition_map import ManifoldPartitionMap, num_samples_per_iteration
from dnn.dataset import Dataset
from commons.ops import calc_avg
import numpy as np

class ReliabilitySpecificManifoldAnalyzer:
    def __init__(self, model, test_data, rel_analyzer) -> None:
        self.model = model
        self.test_data = test_data
        self.rel_analyzer = rel_analyzer
        self.feature_samples = None
        self.rel_scores = None

    def sample(self, gaussian_mixture, num_runs=10):
        result = self.rel_analyzer.analyze_feature_space(self.test_data)
        
        self._print_progress(result.success())

        self.feature_samples = result.X
        self.rel_scores = result.reliability_scores
        for _ in range(num_runs):
            new_feature_samples = gaussian_mixture.sample_features(num_samples_per_iteration)

            dataset = Dataset(X=new_feature_samples, Y=np.zeros(num_samples_per_iteration))
            result = self.rel_analyzer.analyze_feature_space(dataset)
            
            self.feature_samples = np.concatenate((self.feature_samples, new_feature_samples), axis=0)
            self.rel_scores = np.concatenate((self.rel_scores, result.reliability_scores), axis=0)
            
            self._print_progress(calc_avg(self.rel_scores))
    
    def analyze(self, partition_alg):
        partition_map = ManifoldPartitionMap(self.model, partition_alg)
        partition_map.estimate_manifold(self.feature_samples, self.rel_scores)

        return partition_map

    def _print_progress(self, success):
        print("Model: {name}, Success probability: {success}".format(name=self.model.name, success=success))