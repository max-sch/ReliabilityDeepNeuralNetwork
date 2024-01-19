from sklearn import tree
import numpy as np

class ManifoldPartitionMap:
    def __init__(self, bin_resolution, max_depth=10) -> None:
        self.bins = np.arange(0, 1 + bin_resolution, bin_resolution)
        self.partitionMap = tree.DecisionTreeClassifier(max_depth=max_depth)

    def fit(self, features, rel_measures):
        X,Y = self._sort_into_bins(features, rel_measures)
        self.partitionMap.fit(X, Y)

    def _sort_into_bins(self, features, rel_measures):
        X = []
        Y = []

        for feature, rel_measure in zip(features, rel_measures):
            X.append[feature]
            for bin in self.bins:
                if rel_measure <= bin:
                    Y.append(bin)
                    break

        return (X,Y)
