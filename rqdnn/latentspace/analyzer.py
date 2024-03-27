from latentspace.partition_map import ManifoldPartitionMap, num_samples_per_iteration
from dnn.dataset import Dataset
import numpy as np

class ReliabilitySpecificManifoldAnalyzer:
    def __init__(self, model, test_data, rel_analyzer) -> None:
        self.model = model
        self.test_data = test_data
        self.rel_analyzer = rel_analyzer
        self.sampled_rel_scores = []

    def sample(self, gaussian_mixture, num_runs=10):
        result = self.rel_analyzer.analyze_feature_space(self.test_data)
        
        self._print_progress(result.reliability_scores)

        self.sampled_rel_scores = result.reliability_scores
        for _ in range(num_runs):
            feature_samples = gaussian_mixture.sample_features(num_samples_per_iteration)

            dataset = Dataset(X=feature_samples, Y=np.zeros(num_samples_per_iteration))
            result = self.rel_analyzer.analyze_feature_space(dataset)
            
            self.sampled_rel_scores = self.sampled_rel_scores + result.reliability_scores
            self._print_progress(rel_scores=self.sampled_rel_scores)
    
    def analyze(self, partition_alg):
        partition_map = ManifoldPartitionMap(self.model, partition_alg)
        partition_map.estimate_manifold(self.sampled_rel_scores)

        return partition_map

    def _print_progress(self, rel_scores):
        _, scores = list(zip(*rel_scores))
        n = len(scores)
        success = np.matmul(scores, np.transpose(np.ones((n)))) / n
        
        print("Model: {name}, Success probability: {success}".format(name=self.model.name, success=success))