from reliability.conformal_prediction import AdaptiveConformalPrediction

class ReliabilityMeasure:
    def __init__(self, model) -> None:
        self.model = model

    def conditional_success(self, x, y_true):
        '''Determines the conditional success probability of the model's prediction given input x and the true ouput y_true'''
        raise NotImplementedError
    
class ConformalPredictionBasedMeasure(ReliabilityMeasure):
    def __init__(self, model, calibration_set) -> None:
        super().__init__(model)
        self.conformal_predictor = AdaptiveConformalPrediction(model, calibration_set)

    def conditional_success(self, x, y_true):
        norm_confidence = self._normalized_confidence(x, y_true)
        return norm_confidence * (1 - self.conformal_predictor.error_rate)

    def _normalized_confidence(self, x, y_true):
        norm_const = sum(self.model.confidence(x, y) for y in self.conformal_predictor.calc_prediction_set(x))
        if norm_const == 0:
            return 0
        return self.model.confidence(x, y_true) / norm_const



