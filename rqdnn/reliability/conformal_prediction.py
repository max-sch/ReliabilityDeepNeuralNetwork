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
        '''An internal function for calculating the scores.'''
        raise NotImplementedError

    def calc_prediction_set_size(self, softmax):
        '''Calculates the prediction set for softmax outputs'''
    
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
        labels = tuning_data.Y
        labels = labels.reshape((len(labels), 1))

        softmax = model.softmax(tuning_data.X)
        softmax_ranks = np.argsort(softmax, axis=1)[:, ::-1]
        positions = softmax_ranks == labels
        idx_matrix = np.array([np.arange(positions.shape[1])] * positions.shape[0])
        L = idx_matrix[positions]
        
        return np.quantile(L, 1-error_rate, interpolation='higher') + 1

    def _calc_scores(self, calibration_set):
        labels = calibration_set.Y
        softmax = self.model.softmax(calibration_set.X)
        softmax = np.array(softmax)

        desc_ordered_idx = np.argsort(softmax, axis=1)[:,::-1]
        desc_ordered_softmax = np.take_along_axis(softmax, desc_ordered_idx, axis=1)

        reg_vec = np.array(self.k_reg*[0,] + (softmax.shape[1]-self.k_reg)*[self.l_reg,])[None,:]
        reg_softmax = desc_ordered_softmax + reg_vec

        if len(labels.shape) == 1:
            labels = labels[:,None]
        L = np.where(desc_ordered_idx == labels)[1]

        n = calibration_set.size()
        return reg_softmax.cumsum(axis=1)[np.arange(n), L] - np.random.rand(n) * reg_softmax[np.arange(n), L]
    
    def calc_prediction_set_size(self, softmax):
        desc_ordered_idx = np.argsort(softmax, axis=1)[:,::-1]
        softmax = np.array(softmax)
        desc_ordered_softmax = np.take_along_axis(softmax, desc_ordered_idx, axis=1)

        reg_vec = np.array(self.k_reg*[0,] + (softmax.shape[1]-self.k_reg)*[self.l_reg,])[None,:]
        reg_softmax = desc_ordered_softmax + reg_vec

        reg_softmax_cumsum = reg_softmax.cumsum(axis=1)
        indicators = (reg_softmax_cumsum - np.random.rand(1, 1) * reg_softmax) <= self.qhat if self.rand else reg_softmax_cumsum - reg_softmax <= self.qhat
        if self.disallow_zero_sets: 
            indicators[:, 0] = True

        prediction_sets = np.take_along_axis(indicators, desc_ordered_idx.argsort(axis=1), axis=1)
        return np.sum(prediction_sets, axis=1)