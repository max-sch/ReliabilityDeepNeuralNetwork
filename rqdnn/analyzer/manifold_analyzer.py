from analyzer.partition_map import ManifoldPartitionMap

class ReliabilitySpecificManifoldAnalyzer:
    def __init__(self, model, test_data, rel_measure) -> None:
        self.model = model
        self.test_data = test_data
        self.rel_measure = rel_measure

    def analyze(self) -> ManifoldPartitionMap:
        rel_measures,features = self._calc_reliability_measures()

        partition_map = ManifoldPartitionMap(bin_resolution=0.05)
        partition_map.fit(features, rel_measures)

        return partition_map

    def _calc_reliability_measures(self):
        rel_measures = []
        features = []

        for x,y in iter(self.test_data):
            rel_measures.append(self.rel_measure.conditional_success(x, y))
            features.append(self.model.project(x))
        
        return (rel_measures, features)