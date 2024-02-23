from sklearn import tree
from scipy.stats import qmc

import numpy as np

class ManifoldPartitionMap:
    def __init__(self, model) -> None:
        self.model = model
        self.partitioned_space = set()
        self.score_map = {}
        self.estimated_manifold = None

    def estimate_manifold(self, reliability_scores, decision_tree=tree.DecisionTreeClassifier()):
        features, scores = self._prepare_for_analysis(reliability_scores)
        self.estimated_manifold = decision_tree.fit(features, scores)

        print(tree.export_text(self.estimated_manifold))

        leave_nodes = self.estimated_manifold.apply(features)
        for i in range(len(features)):
            leave_node = leave_nodes[i]
            feature = features[i,:]

            partitions = [p for p in self.partitioned_space if p.node_id == leave_node]
            if len(partitions) == 0:
                score_idx = np.argmax(self.estimated_manifold.tree_.value[leave_node])
                partition = Partition(leave_node, self._get_score(score_idx))

                self.partitioned_space.add(partition)
            else:
                partition = partitions[0]

            partition.include(feature)

    def _prepare_for_analysis(self, rel_scores):
        n = len(rel_scores)
        m = len(self.model.project(rel_scores[0][0]))
        features = np.zeros((n, m))
        scores = np.zeros(n, dtype=int)

        score_idx = 0

        for i, (x, score) in enumerate(rel_scores):
            features[i,:] = self.model.project(x)
            
            if score not in self.score_map.keys():
                self.score_map[score] = score_idx
                score_idx += 1
            scores[i] = self.score_map[score]

        return (features, scores)
    
    def _get_score(self, score_idx):
        return [s for s,idx in self.score_map.items() if idx == score_idx][0]
    
class Partition:
    def __init__(self, node_id, rel_score) -> None:
        self.node_id = node_id
        self.rel_score = rel_score
        self.feature_dim_ranges = {}
        self.partitioned_features = []

    def __str__(self) -> str:
        return "Partition with ID {id} and reliability score {score} and acc spans {acc} \n".format(id=self.node_id, 
                                                                                                    score=self.rel_score, 
                                                                                                    acc=self.accumulated_spans())
    
    def include(self, feature):
        self.partitioned_features.append(feature)

        for feature_dim in range(len(feature)):
            feature_dim_val = feature[feature_dim]

            if feature_dim not in self.feature_dim_ranges.keys():
                min_val = max_val = feature_dim_val
            else:
                current_range = self.feature_dim_ranges[feature_dim]
                min_val = feature_dim_val if feature_dim_val < current_range[0] else current_range[0] 
                max_val = feature_dim_val if feature_dim_val > current_range[1] else current_range[1]
            
            self.feature_dim_ranges[feature_dim] = (min_val, max_val)

    def accumulated_spans(self):
        return sum(self._get_spans())
    
    def sample_features(self, num_samples=1000):
        spans = np.array(self._get_spans())
        non_zeros = spans != 0
        non_zero_span_idxs = np.arange(len(spans))[non_zeros]
        zero_span_idxs = np.arange(len(spans))[~non_zeros]
        
        is_singleton = len(spans) == len(zero_span_idxs)
        if is_singleton:
            return np.array([])

        samples = qmc.Halton(d=len(non_zero_span_idxs), scramble=False).random(n=num_samples)
        
        non_zero_ranges = [feature_dim_range for feature_dim, feature_dim_range in self.feature_dim_ranges.items() 
                              if feature_dim in non_zero_span_idxs]
        l_bounds, u_bounds = zip(*non_zero_ranges)
        scaled_samples = qmc.scale(samples, l_bounds, u_bounds)

        num_part_features = len(self.partitioned_features)
        num_rows = num_part_features + num_samples
        num_columns = len(non_zero_span_idxs) + len(zero_span_idxs)
        sampled_features = np.zeros((num_rows, num_columns))
        sampled_features[range(num_part_features),:] = self.partitioned_features

        singletons = [feature_dim_range[0] for feature_dim, feature_dim_range in self.feature_dim_ranges.items() 
                       if feature_dim in zero_span_idxs]
        for i in range(num_part_features, num_part_features + num_samples):
            sample = scaled_samples[i - num_part_features,:]

            enriched_sample = np.zeros(num_columns)
            enriched_sample[zero_span_idxs] = singletons
            enriched_sample[non_zero_span_idxs] = sample

            sampled_features[i,:] = enriched_sample
        
        return sampled_features

    def _get_spans(self):
        return [range[1] - range[0] for _, range in self.feature_dim_ranges.items()]



