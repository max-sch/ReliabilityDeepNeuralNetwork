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
        lower_success_bounds = rel_analyzer.analyze(self.test_data)
        success = sum([p for _,p in lower_success_bounds.items()]) / self.test_data.size()
        print("Success probability: " + str(success))
            
        return None
        #rel_measures,features = self._calc_reliability_measures()

        #partition_map = ManifoldPartitionMap(bin_resolution=0.05)
        #partition_map.fit(features, rel_measures)

        #return partition_map