from reliability.conformal_prediction import RegularizedAdaptiveConformalPrediction
from commons.ops import calc_avg
import numpy as np

class ReliabilityAnalysisResult:
    def __init__(self, model, X, reliability_scores) -> None:
        self.model = model
        self.X = X
        self.reliability_scores = reliability_scores

    def success(self):
        return calc_avg(self.reliability_scores)

class ReliabilityAnalyzer:
    def __init__(self, model) -> None:
        self.model = model

    def analyze_input_space(self, dataset):
        '''Analyzes the reliability the specified dnn given a corresponding dataset of the input space.'''
        raise NotImplementedError
    
    def analyze_feature_space(self, dataset):
        '''Analyzes the reliability the specified dnn given a corresponding dataset of the feature space.'''
        raise NotImplementedError

class ConformalPredictionBasedReliabilityAnalyzer(ReliabilityAnalyzer):
    def __init__(self, 
                 model, 
                 calibration_set, 
                 tuning_set, 
                 step_size=0.01, 
                 class_to_idx_mapper=lambda x:x) -> None:
        self.model = model
        self.calibration_set = calibration_set
        self.tuning_set = tuning_set
        self.step_size = step_size
        self.cached_predictors = {}
        self.class_to_idx_mapper=class_to_idx_mapper
    
    def analyze(self, dataset):
        placeholder = -1
        lower_success_bounds = np.ones((dataset.size())) * placeholder

        for error_level in np.arange(self.step_size, 1 + self.step_size, self.step_size):     
            inputs_to_analyze = lower_success_bounds == placeholder      
            if np.all(~inputs_to_analyze):
                break
            
            _error_level = round(error_level, 2)
            
            if _error_level in self.cached_predictors.keys():
                conf_predictor = self.cached_predictors[_error_level]
            else:
                conf_predictor = RegularizedAdaptiveConformalPrediction(model=self.model,
                                                                        calibration_set=self.calibration_set,
                                                                        tuning_data=self.tuning_set,
                                                                        error_rate=_error_level,
                                                                        class_to_idx_mapper=self.class_to_idx_mapper)
                self.cached_predictors[_error_level] = conf_predictor

            # Calculation of the indices of the inputs (to be analyzed) that have a single prediction set
            X_to_analyze = dataset.X[inputs_to_analyze]
            softmax = self.model.softmax_for_features(X_to_analyze)
            prediction_set_sizes = conf_predictor.calc_prediction_set_size(softmax)
            singletons = prediction_set_sizes == 1
            
            # Set success level of those inputs which prediction set is a singleton
            analyzed_idxs = np.arange(dataset.size(), dtype=int)[inputs_to_analyze] 
            singleton_idxs = analyzed_idxs[singletons]
            success_level = round(1-_error_level, 2)
            lower_success_bounds[singleton_idxs] = np.repeat(a=success_level, repeats=len(singleton_idxs))

        return ReliabilityAnalysisResult(model=self.model,
                                         X=dataset.X,
                                         reliability_scores=lower_success_bounds)