from manifold.partition_map import ManifoldPartitionMap, num_samples
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
        
        print("Number of partitions: {size}".format(size=len(partition_map.partitioned_space)))
        for p in sorted(partition_map.partitioned_space, key=lambda p: p.rel_score, reverse=True):
            print(p)

        sample_size = len(self.test_data)
        for _ in range(10):
            partitions = [p for p in partition_map.partitioned_space if not p.is_singleton()]

            sample_size = (num_samples * len(partitions)) + sample_size
            num_feature_dims = partitions[0].num_feature_dims()
            feature_samples = np.zeros((sample_size, num_feature_dims))
            rel_scores = np.zeros(sample_size)

            current_idx = 0
            for partition in partitions:
                features = partition.partitioned_features
                num_features = len(features)
                feature_samples[range(current_idx, num_features),:] = features
                scores = [partition.rel_score * num_features]
                rel_scores[range(current_idx, num_features)] = scores
                current_idx += num_features

                samples = partition.sample_features()
                scores = [self.model.project(sample) for sample in samples]
                feature_samples[range(current_idx, num_samples),:] = samples
                rel_scores[range(current_idx, num_samples)] = scores
                current_idx += num_samples

            partition_map.restimate_manifold(features=feature_samples, scores=rel_scores)

            print("Number of partitions: {size}".format(size=len(partition_map.partitioned_space)))
            for p in sorted(partition_map.partitioned_space, key=lambda p: p.rel_score, reverse=True):
                print(p)

        return partition_map