from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import numpy as np

num_samples_per_iteration = 10000

class ManifoldPartitionMap:
    def __init__(self, model, partition_alg) -> None:
        self.model = model
        self.score_map = {}
        self.partition_alg = partition_alg

    def calc_scores(self, features):
        score_idxs = self.partition_alg.predict(features)
        return [self._get_score(s) for s in score_idxs]

    def estimate_manifold(self, reliability_scores):
        features, score_idxs = self._prepare_for_feature_space_analysis(reliability_scores)
        self.partition_alg.partition(features, score_idxs)
    
    def _prepare_for_feature_space_analysis(self, rel_scores):
        n = len(rel_scores)
        m = len(rel_scores[0][0])
        features = np.zeros((n, m))
        score_idxs = np.zeros(n, dtype=int)

        if len(self.score_map.values()) == 0:
            score_idx = 0
        else:
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
    
class PartitioningAlgorithm:
    def __init__(self, name) -> None:
        self.name = name

    def predict(self, features):
        '''Predicts the success level for a set of features of the latent space.'''
        raise NotImplementedError

    def partition(self, features, score_idxs):
        '''Main procedure to partition the features accoring to their distance and score (or score index).'''
        raise NotImplementedError
    
class DecisionTreePartitioning(PartitioningAlgorithm):
    def __init__(self) -> None:
        super().__init__("Decision tree pratitioning")
        self.decision_tree = DecisionTreeClassifier()

    def predict(self, features):
        return self.decision_tree.predict(features)
    
    def partition(self, features, score_idxs):
        self.decision_tree.fit(features, score_idxs)

class KnnPartitioning(PartitioningAlgorithm):
    def __init__(self, n_neighbors=5) -> None:
        super().__init__("KNN based partitioning")
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    def predict(self, features):
        return self.knn.predict(features)
    
    def partition(self, features, score_idxs):
        self.knn.fit(features, score_idxs)