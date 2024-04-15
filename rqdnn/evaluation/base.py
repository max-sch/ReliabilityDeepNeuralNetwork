from dnn.dataset import Dataset
from latentspace.analyzer import ReliabilitySpecificManifoldAnalyzer
from commons.print import print_progress, print_start, print_end
from evaluation.metrics import TrueSuccessProbability, AverageReliabilityScores
from evaluation.visual import histoplot, boxplot
from datetime import datetime
from enum import Enum
from os.path import join

import numpy as np
import os

class EvaluationResult:
    def __init__(self, 
                 correct_idxs,
                 incorrect_idxs,
                 evaluation_set, 
                 features, 
                 rel_scores,
                 softmax=None) -> None:
        self.evaluation_set = evaluation_set
        self.correct_idxs = correct_idxs
        self.incorrect_idxs = incorrect_idxs
        self.features = features
        self.rel_scores = rel_scores
        self.softmax = softmax

    def get_correct_features(self):
        return self.features[self.correct_idxs]

    def get_incorrect_features(self):
        return self.features[self.incorrect_idxs]

    def get_correct_scores(self):
        return self.rel_scores[self.correct_idxs]

    def get_incorrect_scores(self):
        return self.rel_scores[self.incorrect_idxs]
    
class EvaluationReport:
    def __init__(self, model_name, result_dir="", format='txt') -> None:
        self.result_dir = result_dir
        self.format = format
        self.date = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        self.content = "Evaluation report {date} for {model}".format(date=self.date, model=model_name)
        self.content = self.content.center(60, '=') + "\n"

    def append(self, result):
        self.content += result + "\n"

    def append_headline(self, headline):
        self.content += headline.center(60, '-') + "\n"

    def save(self):
        self.content += "End of report".center(60, '=')

        file = join(self.result_dir, 'EvaluationReport_{date}.{ext}'.format(date=self.date, ext=self.format))
        result_file = open(file, 'w')
        result_file.write(self.content)
        result_file.close()

class ModelLevel(Enum):
    WORSE = "Worse"
    AVG = "Avg"
    BEST = "Best"

class Evaluation:
    def evaluate(self, 
                 models, 
                 gaussian_cal_set, 
                 evaluation_set,
                 partition_algs, 
                 metrics,
                 include_softmax=False):
        for model in models:
            self.report = EvaluationReport(model.name)

            print_start(model.name)
            self._print_and_report("Calculate reliability scores of {model}".format(model=model.name))

            rel_analyzer = self.create_rel_analyzer_for(model)

            predictions = model.predict_all(evaluation_set.X)
            features = model.project_all(evaluation_set.X)
            rel_scores = calc_rel_scores(features, rel_analyzer)
            softmax = model.softmax(evaluation_set.X) if include_softmax else None
            diffs = predictions - evaluation_set.Y
            result = EvaluationResult(correct_idxs=diffs==0,
                                      incorrect_idxs=diffs!=0, 
                                      evaluation_set=evaluation_set,
                                      features=features, 
                                      rel_scores=rel_scores,
                                      softmax=softmax
            )

            self._evaluate_and_report_metrics(self._load_std_metrics(), result)

            histoplot(scores_correct=result.get_correct_scores(), 
                      scores_incorrect=result.get_incorrect_scores(), 
                      title="Calculated reliability score distribution",
                      show_plot=True)
            boxplot(scores_correct=result.get_correct_scores(),
                    scores_incorrect=result.get_incorrect_scores(),
                    title="Calculated reliability scores (correct and incorrect)",
                    show_plot=True)

            predictions = model.predict_all(gaussian_cal_set.X)
            features = model.project_all(gaussian_cal_set.X)
            gaussian_mixture = self.estimate_gaussian(features, predictions)

            ls_analyzer = ReliabilitySpecificManifoldAnalyzer(model=model, 
                                                              test_data=Dataset(X=features, Y=predictions), 
                                                              rel_analyzer=rel_analyzer
            )
            ls_analyzer.sample(gaussian_mixture)

            for partition_alg in partition_algs:
                self._print_and_report("Estimated reliability scores based on {approach}".format(approach=partition_alg.name))

                partition_map = ls_analyzer.analyze(partition_alg)
                
                estimated_rel_scores = partition_map.calc_scores(result.features)
                result = EvaluationResult(correct_idxs=result.correct_idxs,
                                          incorrect_idxs=result.incorrect_idxs,
                                          evaluation_set=result.evaluation_set,
                                          features=result.features,
                                          rel_scores=estimated_rel_scores,
                                          softmax=result.softmax
                )

                histoplot(scores_correct=result.get_correct_scores(), 
                              scores_incorrect=result.get_incorrect_scores(), 
                              title="Reliability score distribution based on {approach}".format(approach=partition_alg.name),
                              show_plot=True)

                self._evaluate_and_report_metrics(metrics, result)

            self.report.save()
            print_end()

    def estimate_gaussian(self, features, predictions):
        '''Estimates the gaussian mixture for a set of features and predictions of a given dnn model.'''
        raise NotImplementedError
    
    def create_rel_analyzer_for(self, model):
        '''Creates a reliability analyzer for a given DNN model.'''
        raise NotImplementedError
    
    def _load_std_metrics(self):
        return [TrueSuccessProbability(), AverageReliabilityScores()]
    
    def _evaluate_and_report_metrics(self, metrics, result):
        for metric in metrics:
            metric.apply(result)
            metric.print_result()
            if metric.is_visualizable():
                metric.visualize()

            self.report.append(metric.get_report())
    
    def _print_and_report(self, message):
        print_progress(message)
        self.report.append_headline(message)
    

def calc_rel_scores(features, rel_analyzer):
        result = rel_analyzer.analyze_feature_space(Dataset(X=features, Y=np.zeros((len(features)))))
        return result.reliability_scores