from scipy import stats
from commons.ops import calc_avg
from commons.print import print_result
from evaluation.visual import scatterplot

import numpy as np

class Metric:
    def __init__(self, name) -> None:
       self.name = name

    def apply(self, result) -> str:
       '''Derives or calculates a quantitiy of the evaluation result.'''
       raise NotImplementedError

    def print_result(self):
       '''Prints the results; note that the metric must be applied before.'''
       raise NotImplementedError
    
    def is_visualizable(self):
        '''Specifies whether the metric is able to visualize the results.'''

    def visualize(self):
        '''Visualizes the results; note that the metric must be applied before.'''
        raise NotImplementedError

class TrueSuccessProbability(Metric):
    def __init__(self) -> None:
        super().__init__("True success probability")

    def apply(self, result):
        n = len(result.features)
        self.true_success = len(result.get_correct_features()) / n
        
    def print_result(self):
        print_result(metric=self.name, values=str(self.true_success))

    def is_visualizable(self):
        return False

    def visualize(self):
        pass
    
class AverageReliabilityScores(Metric):
    def __init__(self) -> None:
        super().__init__("Average reliability score")

    def apply(self, result):
        self.avg_score_correct = calc_avg(result.get_correct_scores())
        self.avg_score_incorrect = calc_avg(result.get_incorrect_scores())
        self.avg_score = calc_avg(np.concatenate((result.get_correct_scores(), result.get_incorrect_scores()), axis=0))
    
    def print_result(self):
        result = "{val1} (avg), {val2} (avg correct), {val3} (avg incorrect)".format(val1=self.avg_score, 
                                                                                     val2=self.avg_score_correct,
                                                                                     val3=self.avg_score_incorrect)
        print_result(metric=self.name, values=str(result))

    def is_visualizable(self):
        return False
    
    def visualize(self):
        pass
    
class PearsonCorrelation(Metric):
    def __init__(self, determine_deviation) -> None:
        super().__init__("Pearson correlation")
        self.determine_deviation = determine_deviation

    def apply(self, result):
        softmax_incorrect = result.softmax[result.incorrect_idxs]
        labels = result.evaluation_set.Y[result.incorrect_idxs]
        self.output_deviations = self.determine_deviation(softmax_incorrect, labels)

        self.scores_incorrect = result.get_incorrect_scores()

        self.pears_coef = stats.pearsonr(self.output_deviations, self.scores_incorrect)
        
    def print_result(self):
        result = "pvalue: {pvalue}, statistic: {stat}".format(pvalue=self.pears_coef.pvalue,
                                                              stat=self.pears_coef.statistic)
        print_result(metric=self.name, values=result)
        
    def is_visualizable(self):
        return True
    
    def visualize(self):
        scatterplot(scores=self.scores_incorrect,
                    var_compare=self.output_deviations,
                    title="Correlation of incorrect scores with output deviations",
                    title_var_compare="Output deviations",
                    show_plot=True)