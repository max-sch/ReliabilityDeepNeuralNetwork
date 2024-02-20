from manifold.partition_map import ManifoldPartitionMap
from reliability.analyzer import ConformalPredictionBasedReliabilityAnalyzer
from dnn.dataset import MNISTDataset
import numpy as np

class ReliabilitySpecificManifoldAnalyzer:
    def __init__(self, model, test_data, rel_measure, step_size=0.01) -> None:
        self.model = model
        self.test_data = test_data
        self.rel_measure = rel_measure
        self.step_size = step_size

    def analyze(self) -> ManifoldPartitionMap:
        rel_analyzer = ConformalPredictionBasedReliabilityAnalyzer(model=self.model,
                                                                   calibration_set=MNISTDataset.create_cal(),
                                                                   tuning_set=MNISTDataset.create_cal())
        result = rel_analyzer.analyze(self.test_data)

        print("Model: " + self.model.name + ", Success probability: " + str(result.success))
        
        partition_map = ManifoldPartitionMap(model=self.model)
        partition_map.estimate_manifold(result.reliability_scores)

        for partition in sorted(partition_map.partitioned_space, key=lambda p: p.rel_score, reverse=True):
            print(str(partition))

        return partition_map