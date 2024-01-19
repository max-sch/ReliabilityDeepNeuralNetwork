from partition_map import ManifoldPartitionMap

class ReliabilitySpecificManifoldAnalyzer:
    def __init__(self, model, test_data, rel_measure) -> None:
        self.model = model
        self.test_data = test_data
        self.rel_measure = rel_measure

    def analyze(self) -> ManifoldPartitionMap:
        rel_measures,features = self._calc_reliability_measures()

        partition_map = ManifoldPartitionMap()
        partition_map.fit(rel_measures, features)

        return partition_map

    def _calc_reliability_measures(self):
        rel_measures = []
        features = []

        for x,_ in self.test_data:
            rel_measures.append(self.rel_measure.conditional_success(x))
            features.append(self.model.project(x))
        
        return (rel_measures, features)