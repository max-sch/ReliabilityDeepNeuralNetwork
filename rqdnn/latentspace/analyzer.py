from latentspace.partition_map import ManifoldPartitionMap, num_samples_per_iteration
from dnn.dataset import Dataset
from commons.ops import calc_avg
from evaluation.visual import lineplot

import numpy as np

class ReliabilitySpecificManifoldAnalyzer:
    def __init__(self, model, test_data, rel_analyzer) -> None:
        self.model = model
        self.test_data = test_data
        self.rel_analyzer = rel_analyzer
        self.feature_samples = None
        self.rel_scores = None

    def sample(self, gaussian_mixture, num_runs=10):
        result = self.rel_analyzer.analyze(self.test_data)
        
        self._print_progress(result.success())

        self.feature_samples = result.X
        self.rel_scores = result.reliability_scores
        
        convergence_criterion = ThresholdConvergence(result.success())
        while True:
            new_feature_samples = gaussian_mixture.sample_features(num_samples_per_iteration)

            dataset = Dataset(X=new_feature_samples, Y=np.zeros(num_samples_per_iteration))
            result = self.rel_analyzer.analyze(dataset)
            
            self.feature_samples = np.concatenate((self.feature_samples, new_feature_samples), axis=0)
            self.rel_scores = np.concatenate((self.rel_scores, result.reliability_scores), axis=0)
            
            success = calc_avg(self.rel_scores)
            self._print_progress(success)

            if convergence_criterion.is_satisfied(success):
                break

        convergence_criterion.print_convergence()
    
    def analyze(self, partition_alg):
        partition_map = ManifoldPartitionMap(partition_alg)
        partition_map.estimate_manifold(self.feature_samples, self.rel_scores)

        return partition_map

    def _print_progress(self, success):
        print("Model: {name}, Success probability: {success}".format(name=self.model.name, success=success))

class ConvergenceCriterion:
    def __init__(self, success_prob) -> None:
        self.num_runs = 0
        self.success_probs = [success_prob]

    def is_satisfied(self, current):
        self.success_probs.append(current)
        self.num_runs += 1

        return self.check_criterion()
    
    def check_criterion(self) -> bool:
        '''This method evaluates whether the criterion is met'''
        raise NotImplementedError
    
    def print_convergence(self):
        lineplot(num_runs=[*range(self.num_runs + 1)], scores=self.success_probs)
        
class MaxRunsAreReached(ConvergenceCriterion):
    def __init__(self, success_prob, max_runs) -> None:
        super().__init__(success_prob)
        self.max_runs = max_runs        

    def check_criterion(self) -> bool:
        return True if self.max_runs == self.num_runs else False
    
class ThresholdConvergence(ConvergenceCriterion):
    def __init__(self, success_prob, threshold=0.0001) -> None:
        super().__init__(success_prob)
        self.threshold = threshold

    def check_criterion(self) -> bool:
        n = len(self.success_probs) - 1
        diff = abs(self.success_probs[n] - self.success_probs[n-1])
        return diff <= self.threshold
