import numpy as np

class Metric:
   def __init__(self, name) -> None:
       self.name = name

   def apply(self, evaluation_result) -> str:
       '''Derives or calculates a quantitiy of the evaluation result.'''

class TrueSuccessProbability(Metric):
    def __init__(self) -> None:
        super().__init__("True success probability")

    def apply(self, evaluation_result):
        n = len(evaluation_result.diffs)
        true_success = (n - np.count_nonzero(evaluation_result.diffs)) / n
        return str(true_success)
    
class AverageReliabilityScores(Metric):
    def __init__(self) -> None:
        super().__init__("Average reliability score")

    def apply(self, evaluation_result):
        def calc_avg(scores):
            return np.matmul(np.transpose(scores), np.ones((len(scores)))) / len(scores)

        avg_scores_correct = calc_avg(evaluation_result.scores_correct)
        avg_scores_incorrect = calc_avg(evaluation_result.scores_incorrect)
        
        return "{val1} (avg correct), {val2} (avg incorrect)".format(val1=avg_scores_correct, 
                                                                     val2=avg_scores_incorrect)