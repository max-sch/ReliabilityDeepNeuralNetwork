from reliability.conformal_prediction import RegularizedAdaptiveConformalPrediction
import numpy as np

class ReliabilityAnalyzer:
    def __init__(self, model) -> None:
        self.model = model

    def analyze(self, dataset):
        '''Analyzes the reliability the specified dnn given a corresponding dataset.'''

class ConformalPredictionBasedReliabilityAnalyzer(ReliabilityAnalyzer):
    def __init__(self, model, calibration_set, tuning_set, step_size=0.01) -> None:
        self.model = model
        self.calibration_set = calibration_set
        self.tuning_set = tuning_set
        self.step_size = step_size

    def analyze(self, dataset):
        enum_dataset = {k:v for k, (v,_) in enumerate(dataset)}
        lower_success_bounds = {}
        cached_keys = []

        for error_rate in np.arange(self.step_size, 1 + self.step_size, self.step_size):
            if len(enum_dataset) == 0:
                return lower_success_bounds
            
            print("Error rate: " + str(error_rate))
            conf_predictor = RegularizedAdaptiveConformalPrediction(model=self.model,
                                                                    calibration_set=self.calibration_set,
                                                                    tuning_data=self.tuning_set,
                                                                    error_rate=error_rate)

            for i,x in enum_dataset.items():
                pred_set = conf_predictor.calc_prediction_set(x)
                if (len(pred_set) == 1) and (i not in lower_success_bounds.keys()):
                    lower_success_bounds[i] = 1 - error_rate
                    cached_keys.append(i)

            enum_dataset = {k:v for k,v in enum_dataset.items() if k not in cached_keys}
            
        return lower_success_bounds