import numpy as np

def sort_descending(softmax):
        return [k for k,_ in sorted(softmax.items(), key=lambda item: item[1], reverse=True)] 

class ConformalPrediction:
    def __init__(self, model, calibration_set, error_rate=0.1, class_to_idx_mapper=lambda x:x) -> None:
        self.model = model
        self.error_rate = error_rate
        self.class_to_idx_mapper = class_to_idx_mapper
        self.qhat = self._calc_qhat(calibration_set, error_rate)

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
    def __init__(self, model, calibration_set, error_rate=0.1, class_to_idx_mapper=lambda x:x) -> None:
        super().__init__(model, calibration_set, error_rate, class_to_idx_mapper)

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
    def __init__(self, model, calibration_set, error_rate=0.1, class_to_idx_mapper=lambda x:x) -> None:
        super().__init__(model, calibration_set, error_rate, class_to_idx_mapper)

    def _calc_scores(self, calibration_set):
        scores = []
        for x,label in calibration_set:
            softmax = self.model.get_confidences(x)
            score = sum(softmax[y] for y in sort_descending(softmax) if y == label)
            scores.append(score)
        return scores
    
    def calc_prediction_set(self, x):
        softmax = self.model.get_confidences(x)
        
        prediction_set = set()
        acc_scores = 0
        for y in sort_descending(softmax):
            prediction_set.add(y)

            acc_scores += softmax[y]
            if acc_scores > self.qhat:
                break            

        return prediction_set
    
class RegularizedAdaptiveConformalPrediction(ConformalPrediction):
    def __init__(self, 
                 model, 
                 calibration_set, 
                 tuning_data, 
                 error_rate=0.1,
                 class_to_idx_mapper=lambda x:x, 
                 l_reg=0.01, 
                 rand=True, 
                 disallow_zero_sets=False) -> None:
        self.l_reg = l_reg
        self.rand = rand
        self.k_reg = self._calc_k_reg(tuning_data, model, error_rate)
        self.disallow_zero_sets = disallow_zero_sets
        super().__init__(model, calibration_set, error_rate, class_to_idx_mapper)

    def _calc_k_reg(self, tuning_data, model, error_rate):
        L = []
        for x,label in tuning_data:
            softmax = model.get_confidences(x)
            for i,y in enumerate(sort_descending(softmax)): 
                if y == label:
                    L.append(i + 1)
        
        n = tuning_data.size()
        q_level = np.ceil((n+1)*(1-error_rate))/n

        return np.quantile(L, q_level, interpolation='higher')

    def _calc_scores(self, calibration_set):
        n = calibration_set.size()
        m = self.model.get_output_shape()
        softmax = np.zeros((n,m))
        
        for i, (x,_) in enumerate(calibration_set): 
            for c,p in self.model.get_confidences(x).items():
                softmax[i, self.class_to_idx_mapper(c)] = p
        
        labels = calibration_set.Y
        
        desc_ordered_idx = softmax.argsort(1)[:,::-1] 
        desc_ordered_softmax = np.take_along_axis(softmax, desc_ordered_idx, axis=1)

        reg_vec = np.array(self.k_reg*[0,] + (softmax.shape[1]-self.k_reg)*[self.l_reg,])[None,:]
        reg_softmax = desc_ordered_softmax + reg_vec

        L = np.where(desc_ordered_idx == labels[:,None])[1]

        return reg_softmax.cumsum(axis=1)[np.arange(n), L] - np.random.rand(n) * reg_softmax[np.arange(n), L]
    
    def calc_prediction_set(self, x):
        prediction = self.model.get_confidences(x)
        softmax = np.zeros((1, len(prediction)))

        for c,p in prediction.items():
            softmax[0, self.class_to_idx_mapper(c)] = p
        
        desc_ordered_idx = softmax.argsort(1)[:,::-1]
        desc_ordered_softmax = np.take_along_axis(softmax, desc_ordered_idx, axis=1)

        reg_vec = np.array(self.k_reg*[0,] + (softmax.shape[1]-self.k_reg)*[self.l_reg,])[None,:]
        reg_softmax = desc_ordered_softmax + reg_vec

        reg_softmax_cumsum = reg_softmax.cumsum(axis=1)
        indicators = (reg_softmax_cumsum - np.random.rand(1, 1) * reg_softmax) <= self.qhat if self.rand else reg_softmax_cumsum - reg_softmax <= self.qhat
        if self.disallow_zero_sets: 
            indicators[:, 0] = True

        prediction_set = np.take_along_axis(indicators, desc_ordered_idx.argsort(axis=1), axis=1)
        return softmax[prediction_set]