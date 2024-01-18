class ReliabilityMeasure:
    def __init__(self, model) -> None:
        self.model = model

    def success_prob_given(self, x):
        '''Determines the conditional success probability of the model's prediction given input x'''
        raise NotImplementedError
    
class ConformalPredictionBasedMeasure(ReliabilityMeasure):
    def __init__(self, model, calibration_set) -> None:
        super().__init__(model)
        self.calibration_set = calibration_set

    def success_prob_given(self, x):
        raise NotImplementedError
