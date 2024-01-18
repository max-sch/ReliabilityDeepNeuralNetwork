class Result:
    pass

class ReliabilitySpecificManifoldAnalyzer:
    def __init__(self, model, test_data, rel_measure) -> None:
        self.model = model
        self.test_data = test_data
        self.rel_measure = rel_measure

    def analyze(self) -> Result:
        (rel_measures, features) = self._calc_reliability_measures()

        return Result()

    def _calc_reliability_measures(self):
        rel_measures = []
        features = []

        for x,_ in self.test_data:
            rel_measures.append(self.rel_measure.success_prob_given(x))
            features.append(self.model.project(x))
        
        return (rel_measures, features)