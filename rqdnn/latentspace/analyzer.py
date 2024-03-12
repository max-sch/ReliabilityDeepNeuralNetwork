from latentspace.partition_map import ManifoldPartitionMap, num_samples_per_iteration
from reliability.analyzer import ConformalPredictionBasedReliabilityAnalyzer
from dnn.dataset import MNISTDataset, Dataset
import numpy as np
import math

class ReliabilitySpecificManifoldAnalyzer:
    def __init__(self, model, test_data, step_size=0.01) -> None:
        self.model = model
        self.test_data = test_data
        self.rel_analyzer = ConformalPredictionBasedReliabilityAnalyzer(model=self.model,
                                                                        calibration_set=MNISTDataset.create_randomly(),
                                                                        tuning_set=MNISTDataset.create_randomly())
        self.step_size = step_size

    def analyze(self) -> ManifoldPartitionMap:
        result = self.rel_analyzer.analyze_input_space(self.test_data)
        
        partition_map = ManifoldPartitionMap(model=self.model)
        partition_map.estimate_manifold(result.reliability_scores)
        
        self._print_progress(success=result.success, partition_map=partition_map)

        self._sample_until_convergence(partition_map)

        return partition_map

    def _sample_until_convergence(self, partition_map):
        num_feature_dims = partition_map.num_feature_dims()
        
        for _ in range(10):
            feature_samples = np.zeros((num_samples_per_iteration, num_feature_dims))
            
            non_singletons = [(p, p.accumulated_spans()) for p in partition_map.partitioned_space if not p.is_singleton()]
            overall_acc_spans = sum(p[1] for p in non_singletons)

            current_idx = 0
            for partition,acc_spans in sorted(non_singletons, key=lambda p:p[1], reverse=True):
                if current_idx >= num_samples_per_iteration:
                    break

                num_samples = math.ceil((acc_spans / overall_acc_spans) * num_samples_per_iteration)
                if num_samples + current_idx > num_samples_per_iteration:
                    num_samples = num_samples_per_iteration - current_idx

                samples = partition.sample_features(num_samples=num_samples)
                feature_samples[range(current_idx, num_samples + current_idx),:] = samples
                current_idx += num_samples

            dataset = Dataset(X=feature_samples, Y=np.zeros(num_samples_per_iteration))
            result = self.rel_analyzer.analyze_feature_space(dataset)
            
            old_rel_scores = [(f, p.rel_score) for p in partition_map.partitioned_space for f in p.partitioned_features]
            new_rel_scores = result.reliability_scores + old_rel_scores
            partition_map.reestimate_manifold(reliability_scores=new_rel_scores)

            self._print_progress(success=result.success, partition_map=partition_map)

    def _print_progress(self, success, partition_map):
        print("Model: " + self.model.name + ", Success probability: {success}".format(success=success))
        print("Number of partitions: {size}".format(size=len(partition_map.partitioned_space)))
        acc = sum(p.accumulated_spans() for p in partition_map.partitioned_space)
        print("Overall acc spans: {acc}".format(acc=acc))