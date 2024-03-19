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

    def analyze_input_space(self, dataset):
        '''Analyzes the reliability the specified dnn given a corresponding dataset of the input space.'''
        raise NotImplementedError
    
    def analyze_feature_space(self, dataset):
        '''Analyzes the reliability the specified dnn given a corresponding dataset of the feature space.'''
        raise NotImplementedError

class ConformalPredictionBasedReliabilityAnalyzer(ReliabilityAnalyzer):
    def __init__(self, model, calibration_set, tuning_set, step_size=0.01) -> None:
        self.model = model
        self.calibration_set = calibration_set
        self.tuning_set = tuning_set
        self.step_size = step_size
        self.cached_predictors = {}

    def analyze_input_space(self, dataset):
        return self._analyze(dataset=dataset, conf_predictor_switch=lambda conf_predictor, x: conf_predictor.calc_prediction_set(x))
    
    def analyze_feature_space(self, dataset):
        def conf_predictor_switch(conf_predictor, feature):
            adj_feature = np.zeros((1, len(feature)))
            adj_feature[0,:] = feature
            softmax = self.model.get_confidences_for_feature(adj_feature)
            return conf_predictor.calc_prediction_set_for_confidences(softmax)
        return self._analyze(dataset=dataset, conf_predictor_switch=conf_predictor_switch)

    def _analyze(self, dataset, conf_predictor_switch):
        enum_dataset = {k:v for k, (v,_) in enumerate(dataset)}
        lower_success_bounds = {}
        cached_keys = []

        for error_level in np.arange(self.step_size, 1 + self.step_size, self.step_size):           
            if len(enum_dataset) == 0:
                break
            
            adj_error_level = round(error_level, 2)
            #print("Error rate: " + str(adj_error_level))
            
            if adj_error_level in self.cached_predictors.keys():
                conf_predictor = self.cached_predictors[adj_error_level]
            else:
                conf_predictor = RegularizedAdaptiveConformalPrediction(model=self.model,
                                                                        calibration_set=self.calibration_set,
                                                                        tuning_data=self.tuning_set,
                                                                        error_rate=adj_error_level)
                self.cached_predictors[adj_error_level] = conf_predictor

            for i,x in enum_dataset.items():
                pred_set = conf_predictor_switch(conf_predictor, x)
                if (len(pred_set) == 1) and (i not in lower_success_bounds.keys()):
                    success_level = round(1 - adj_error_level, 2)
                    lower_success_bounds[i] = (x, success_level)
                    cached_keys.append(i)

            enum_dataset = {k:v for k,v in enum_dataset.items() if k not in cached_keys}

        rel_scores = list(lower_success_bounds.values())
        return ReliabilityAnalysisResult(model=self.model,
                                         success=sum([s for _,s in rel_scores]) / dataset.size(),
                                         reliability_scores=rel_scores)