import numpy as np

class ConformalPrediction:
    def __init__(self, model, calibration_set, error_rate=0.1) -> None:
        self.model = model
        self.qhat = self._calc_qhat(calibration_set, error_rate)
        self.error_rate = error_rate

    def _calc_qhat(self, calibration_set, error_rate):
        conformal_scores = [1 - self.model.confidence(x,y) for x,y in iter(calibration_set)]
        
        n = calibration_set.size()
        q_level = np.ceil((n+1)*(1-error_rate))/n

        return np.quantile(conformal_scores, q_level, interpolation='higher')

    def calc_prediction_set(self, x):
        return {y for y,confidence in self.get_confidences(x).items() if confidence >= 1-self.qhat}