import numpy as np

class ConformalPrediction:
    def __init__(self, model, calibration_set, error_rate=0.1) -> None:
        self.model = model
        self.qhat = self._calc_qhat(calibration_set, error_rate)
        self.error_rate = error_rate

    def _calc_qhat(self, calibration_set, error_rate):
        conformal_scores = self._calc_scores(calibration_set)
        
        n = calibration_set.size()
        q_level = np.ceil((n+1)*(1-error_rate))/n

        return np.quantile(conformal_scores, q_level, interpolation='higher')
    
    def _calc_scores(self, calibration_set):
        raise NotImplementedError

    def calc_prediction_set(self, x):
        raise NotImplementedError
    
class DefaultConformalPrediction(ConformalPrediction):
    def __init__(self, model, calibration_set, error_rate=0.1) -> None:
        super().__init__(model, calibration_set, error_rate)

    def _calc_scores(self, calibration_set):
        return [1 - self.model.confidence(x,y) for x,y in iter(calibration_set)]
    
    def calc_prediction_set(self, x):
        result = set() 
        confidences = self.model.get_confidences(x)
        for y,confidence in confidences.items():
            if confidence >= 1-self.qhat:
                result.add(y)
        if len(result) == 0:
            print("Another one")
        
        return result
        #return {y for y,confidence in self.model.get_confidences(x).items() if confidence >= 1-self.qhat}
    
class AdaptiveConformalPrediction(ConformalPrediction):
    def __init__(self, model, calibration_set, error_rate=0.1) -> None:
        super().__init__(model, calibration_set, error_rate)

    def _calc_scores(self, calibration_set):
        scores = []
        for x,label in calibration_set:
            softmax = self.model.get_confidences(x)
            score = sum(softmax[y] for y in self._sort_descending(softmax) if y == label)
            scores.append(score)
        return scores
    
    def calc_prediction_set(self, x):
        softmax = self.model.get_confidences(x)
        
        prediction_set = set()
        acc_scores = 0
        for y in self._sort_descending(softmax):
            prediction_set.add(y)

            acc_scores += softmax[y]
            if acc_scores > self.qhat:
                break            
        
        if len(prediction_set) == 0:
            print("+++++++++++++++++++++++++++++")

        return prediction_set
    
    def _sort_descending(self, softmax):
        return [k for k,_ in sorted(softmax.items(), key=lambda item: item[1], reverse=True)] 