from sklearn import tree
import numpy as np

class ManifoldPartitionMap:
    def __init__(self, model) -> None:
        self.model = model
        self.partitionMap = set()
        self.score_map = {}
        self.approx_manifold = None

    def approximate(self, reliability_scores, decision_tree=tree.DecisionTreeClassifier()):
        features, scores = self._prepare_for_analysis(reliability_scores)
        self.approx_manifold = decision_tree.fit(features, scores)

        print(tree.export_text(self.approx_manifold))

        decision_paths = self.approx_manifold.decision_path(features)
        leave_nodes = self.approx_manifold.apply(features)
        n = len(features)
        for i in range(n):
            leave_node = leave_nodes[i]

            partitions = [p for p in self.partitionMap if p.node_id == leave_node]
            if len(partitions) == 0:
                score_idx = np.argmax(self.approx_manifold.tree_.value[leave_node])
                partition = Partition(leave_node, score_idx)

                self.partitionMap.add(partition)
            else:
                partition = partitions[0]

            nodes = decision_paths.indices[decision_paths.indptr[i] : decision_paths.indptr[i + 1]]
            for node in nodes:
                if leave_node == node:
                    continue

                feature = self.approx_manifold.tree_.feature[node]
                threshold = self.approx_manifold.tree_.threshold[node]
                if features[i, feature] <= threshold:
                    max_val = threshold
                    min_val = features[i, feature]
                else:
                    max_val = features[i, feature]
                    min_val = threshold

                partition.add_or_update_interval(feature, min_val, max_val)


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
    
class Partition:
    def __init__(self, node_id, score_idx) -> None:
        self.node_id = node_id
        self.score_idx = score_idx 
        self.feature_interval = {}

    def add_or_update_interval(self, feature, min_val, max_val):
        if feature not in self.feature_interval.keys():
            self.feature_interval[feature] = (min_val, max_val)
        else:
            current = self.feature_interval[feature]
            new_min_val = min_val if min_val < current[0] else current[0] 
            new_max_val = max_val if max_val > current[1] else current[1]
            
            self.feature_interval[feature] = (new_min_val, new_max_val)


