from reliability.conformal_prediction import RegularizedAdaptiveConformalPrediction
import numpy as np

class ReliabilityAnalysisResult:
    def __init__(self, model, success, reliability_scores) -> None:
        self.model = model
        self.success = success
        self.reliability_scores = reliability_scores


class ReliabilityAnalyzer:
    def __init__(self, model) -> None:
        self.model = model

    def analyze(self, dataset):
        '''Analyzes the reliability the specified dnn given a corresponding dataset.'''
        raise NotImplementedError

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

        for error_level in np.arange(self.step_size, 1 + self.step_size, self.step_size):           
            if len(enum_dataset) == 0:
                break
            
            adj_error_level = round(error_level, 2)
            print("Error rate: " + str(adj_error_level))
            
            conf_predictor = RegularizedAdaptiveConformalPrediction(model=self.model,
                                                                    calibration_set=self.calibration_set,
                                                                    tuning_data=self.tuning_set,
                                                                    error_rate=adj_error_level)

            for i,x in enum_dataset.items():
                pred_set = conf_predictor.calc_prediction_set(x)
                if (len(pred_set) == 1) and (i not in lower_success_bounds.keys()):
                    success_level = round(1 - adj_error_level, 2)
                    lower_success_bounds[i] = (x, success_level)
                    cached_keys.append(i)

            enum_dataset = {k:v for k,v in enum_dataset.items() if k not in cached_keys}

        rel_scores = list(lower_success_bounds.values())
        return ReliabilityAnalysisResult(model=self.model,
                                         success=sum([s for _,s in rel_scores]) / dataset.size(),
                                         reliability_scores=rel_scores)