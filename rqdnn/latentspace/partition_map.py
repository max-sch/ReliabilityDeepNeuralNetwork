from sklearn import tree
from scipy.stats import qmc
from scipy.spatial import ConvexHull
from numpy.linalg import norm

import numpy as np

num_samples_per_iteration = 1000

class ManifoldPartitionMap:
    def __init__(self, model) -> None:
        self.model = model
        self.partitioned_space = set()
        self.score_map = {}
        self.decision_tree = tree.DecisionTreeClassifier()

    def num_feature_dims(self):
        try:
            any = next(iter(self.partitioned_space))
            return any.num_feature_dims()
        except StopIteration:
            return 0

    def estimate_manifold(self, reliability_scores):
        features, score_idxs = self._prepare_for_input_space_analysis(reliability_scores)
        self._estimate_manifold(features=features, score_idxs=score_idxs)

    def reestimate_manifold(self, reliability_scores):
        self.decision_tree = tree.DecisionTreeClassifier()
        self.partitioned_space = set()

        features, score_idxs = self._prepare_for_feature_space_analysis(reliability_scores)
        self._estimate_manifold(features=features, score_idxs=score_idxs)

    def _estimate_manifold(self, features, score_idxs):
        self.decision_tree.fit(features, score_idxs)

        #print(tree.export_text(self.decision_tree))

        leave_nodes = self.decision_tree.apply(features)
        for i in range(len(features)):
            leave_node = leave_nodes[i]
            feature = features[i,:]

            partitions = [p for p in self.partitioned_space if p.node_id == leave_node]
            if len(partitions) == 0:
                score_idx = np.argmax(self.decision_tree.tree_.value[leave_node])
                partition = Partition(leave_node, self._get_score(score_idx))

                self.partitioned_space.add(partition)
            else:
                partition = partitions[0]

            partition.include(feature)

    def _prepare_for_input_space_analysis(self, rel_scores):
        n = len(rel_scores)
        m = len(self.model.project(rel_scores[0][0]))
        features = np.zeros((n, m))
        score_idxs = np.zeros(n, dtype=int)

        score_idx = 0

        for i, (x, score) in enumerate(rel_scores):
            features[i,:] = self.model.project(x)
            
            if score not in self.score_map.keys():
                self.score_map[score] = score_idx
                score_idx += 1
            score_idxs[i] = self.score_map[score]

        return (features, score_idxs)
    
    def _prepare_for_feature_space_analysis(self, rel_scores):
        n = len(rel_scores)
        m = len(rel_scores[0][0])
        features = np.zeros((n, m))
        score_idxs = np.zeros(n, dtype=int)

        score_idx = max(self.score_map.values()) + 1

        for i, (x, score) in enumerate(rel_scores):
            features[i,:] = x
            
            if score not in self.score_map.keys():
                self.score_map[score] = score_idx
                score_idx += 1
            score_idxs[i] = self.score_map[score]

        return (features, score_idxs)
    
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

    def is_singleton(self):
        spans = np.array(self._get_spans())
        return np.all(spans == 0)
    
    def num_feature_dims(self):
        return len(self.feature_dim_ranges)
    
    def accumulated_spans(self):
        return sum(self._get_spans())
    
    def sample_features(self, num_samples):
        spans = np.array(self._get_spans())
        non_zeros = spans != 0
        non_zero_span_idxs = np.arange(len(spans))[non_zeros]
        zero_span_idxs = np.arange(len(spans))[~non_zeros]

        samples = qmc.Halton(d=len(non_zero_span_idxs), scramble=False).random(n=num_samples)
        
        non_zero_ranges = [feature_dim_range for feature_dim, feature_dim_range in self.feature_dim_ranges.items() 
                              if feature_dim in non_zero_span_idxs]
        l_bounds, u_bounds = zip(*non_zero_ranges)
        scaled_samples = qmc.scale(samples, l_bounds, u_bounds)

        num_dims = len(non_zero_span_idxs) + len(zero_span_idxs)
        sampled_features = np.zeros((num_samples, num_dims))

        singletons = [feature_dim_range[0] for feature_dim, feature_dim_range in self.feature_dim_ranges.items() 
                       if feature_dim in zero_span_idxs]
        for i in range(0, num_samples):
            sample = scaled_samples[i,:]

            enriched_sample = np.zeros(num_dims)
            enriched_sample[zero_span_idxs] = singletons
            enriched_sample[non_zero_span_idxs] = sample

            sampled_features[i,:] = enriched_sample
        
        return sampled_features

    def _get_spans(self):
        return [range[1] - range[0] for _, range in self.feature_dim_ranges.items()]
    
class ConvexHullPartition(Partition):
    def __init__(self, node_id, rel_score) -> None:
        super().__init__(node_id, rel_score)

    def __str__(self) -> str:
        return "Partition with ID {id} and reliability score {score} and volume {vol} \n".format(id=self.node_id, 
                                                                                                 score=self.rel_score, 
                                                                                                 vol=self.volume())

    def is_singleton(self):
        return len(self.partitioned_features) == 1
    
    def num_feature_dims(self):
        return len(self.partitioned_features[0]) if len(self.partitioned_features) != 0 else 0
    
    def volume(self):
        if self.is_singleton():
            return 0
        spans = np.array(self._get_spans())
        non_zeros = spans != 0
        non_zero_span_idxs = np.arange(len(spans))[non_zeros]
        sub_feature_space = np.array(self.partitioned_features)[:,non_zero_span_idxs]
        return ConvexHull(points=sub_feature_space).volume
    
    def sample_features(self, num_samples):
        spans = np.array(self._get_spans())
        non_zeros = spans != 0
        non_zero_span_idxs = np.arange(len(spans))[non_zeros]
        zero_span_idxs = np.arange(len(spans))[~non_zeros]

        samples = qmc.Halton(d=len(non_zero_span_idxs), scramble=False).random(n=num_samples)
        
        non_zero_ranges = [feature_dim_range for feature_dim, feature_dim_range in self.feature_dim_ranges.items() 
                              if feature_dim in non_zero_span_idxs]
        l_bounds, u_bounds = zip(*non_zero_ranges)
        scaled_samples = qmc.scale(samples, l_bounds, u_bounds)

        sub_feature_space = np.array(self.partitioned_features)[:,non_zero_span_idxs]
        in_hull_samples = self.filter_in_hull_samples(sub_feature_space, scaled_samples)

        num_dims = len(non_zero_span_idxs) + len(zero_span_idxs)
        sampled_features = np.zeros((num_samples, num_dims))

        singletons = [feature_dim_range[0] for feature_dim, feature_dim_range in self.feature_dim_ranges.items() 
                       if feature_dim in zero_span_idxs]
        for i in range(0, num_samples):
            sample = in_hull_samples[i,:]

            enriched_sample = np.zeros(num_dims)
            enriched_sample[zero_span_idxs] = singletons
            enriched_sample[non_zero_span_idxs] = sample

            sampled_features[i,:] = enriched_sample
        
        return sampled_features

    def filter_in_hull_samples(self, features, samples, tolerance=1e-12):
        hull = ConvexHull(features)
        
        n = hull.equations.shape[1] - 1
        A = np.delete(hull.equations, n, axis=1)
        b = hull.equations[:,n]

        result = np.matmul(samples, np.transpose(A)) + b
        samples_in_hull_idxs = np.all(result <= tolerance, axis=1)
        return samples[samples_in_hull_idxs,:]



