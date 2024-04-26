from scipy import stats
from commons.ops import calc_avg
from commons.print import print_result
from evaluation.visual import scatterplot, barplot

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
    
    def get_report(self):
        '''Returns the string formatted result.'''
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

    def get_report(self):
        return "{name}: {result}".format(name=self.name, result=self.true_success)

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
        print_result(metric=self.name, values=self._to_string())

    def get_report(self):
        return "{name}: {result}".format(name=self.name, result=self._to_string())

    def is_visualizable(self):
        return False
    
    def visualize(self):
        pass

    def _to_string(self):
        return "{val1} (avg), {val2} (avg correct), {val3} (avg incorrect)".format(val1=self.avg_score, 
                                                                                   val2=self.avg_score_correct,
                                                                                   val3=self.avg_score_incorrect)

class AverageOutputDeviation(Metric):
    def __init__(self, determine_deviation) -> None:
        super().__init__("Average output deviation")
        self.determine_deviation = determine_deviation

    def apply(self, result):
        softmax_incorrect = result.softmax[result.incorrect_idxs]
        labels = result.evaluation_set.Y[result.incorrect_idxs]
        out_deviations = self.determine_deviation(softmax_incorrect, labels)
        self.avg_output_deviation = calc_avg(out_deviations)
        
    def print_result(self):
        print_result(metric=self.name, values=str(self.avg_output_deviation))

    def get_report(self):
        return "{name}: {result}".format(name=self.name, result=self.avg_output_deviation)

    def is_visualizable(self):
        return False
    
    def visualize(self):
        pass

class SoftmaxPositionToReliabilityCorrelation(Metric):
    def __init__(self, determine_deviation, num_pos) -> None:
        super().__init__("Softmax position to reliability correlation")
        self.determine_deviation = determine_deviation
        self.num_pos = num_pos
        self.avg_scores = np.zeros((num_pos))
        self.num_samples_per_pos = np.zeros((num_pos))

    def apply(self, result) -> str:
        self.out_deviations = self.determine_deviation(result.softmax, result.evaluation_set.Y)
        self.rel_scores = result.rel_scores

        for pos in range(self.num_pos):
            scores = self.rel_scores[self.out_deviations == pos]
            self.avg_scores[pos] = calc_avg(scores)
            self.num_samples_per_pos[pos] = len(scores)

    def print_result(self):
        print_result(metric=self.name, values=self._to_string())

    def get_report(self):
        return "{name}: {result}".format(name=self.name, result=self._to_string())

    def is_visualizable(self):
        return True
    
    def visualize(self):
        avg_scores = []
        num_samples = []
        for pos in range(self.num_pos):
            samples = self.num_samples_per_pos[pos]
            if samples >= 50:
                avg_scores.append(self.avg_scores[pos])
                num_samples.append(int(samples))
            else:
                avg_scores.append(0)
                num_samples.append(0)

        barplot(avg_scores, range(self.num_pos), num_samples, show_plot=True)
        
    def _to_string(self):
        str_vals = [(pos,self.avg_scores[pos],self.num_samples_per_pos[pos]) for pos in range(self.num_pos)]
        return ','.join('(pos: {}, score: {}, #samples: {})'.format(val[0], val[1], val[2]) for val in str_vals)

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
        print_result(metric=self.name, values=self._to_string())

    def get_report(self):
        return "{name}: {result}".format(name=self.name, result=self._to_string())
        
    def is_visualizable(self):
        return True
    
    def visualize(self):
        scatterplot(scores=self.scores_incorrect,
                    var_compare=self.output_deviations,
                    title="Correlation of incorrect scores with output deviations",
                    title_var_compare="Output deviations",
                    show_plot=True)
        
    def _to_string(self):
        return "pvalue: {pvalue}, statistic: {stat}".format(pvalue=self.pears_coef.pvalue,
                                                            stat=self.pears_coef.statistic)