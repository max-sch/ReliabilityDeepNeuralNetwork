from sklearn import tree
import numpy as np

class ManifoldPartitionMap:
    def __init__(self, model, max_depth=10) -> None:
        self.model = model
        self.partitionMap = tree.DecisionTreeClassifier(max_depth=max_depth)
        self.score_map = {}

    def fit(self, reliability_scores):
        features, scores = self._prepare_for_analysis(reliability_scores)
        self.partitionMap.fit(features, scores)

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
