from dnn.dataset import Dataset
from latentspace.analyzer import ReliabilitySpecificManifoldAnalyzer
from commons.print import print_progress, print_start, print_end
from evaluation.metrics import TrueSuccessProbability, AverageReliabilityScores
from enum import Enum
from evaluation.report import EvaluationReport, ModelReport

import numpy as np

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

class ModelLevel(Enum):
    WORST = "Worst"
    AVG = "Avg"
    BEST = "Best"

class Evaluation:
    def evaluate(self, 
                 models, 
                 gaussian_cal_set, 
                 evaluation_set,
                 partition_algs, 
                 metrics,
                 include_softmax=False,
                 disable_partitioning=True,
                 repetitions=5):
        reports = []
        for model in models:
            report = EvaluationReport(model)   

            for repetition in range(repetitions):
                model_report = self._evaluate(model,
                                              gaussian_cal_set,
                                              evaluation_set,
                                              partition_algs,
                                              metrics,
                                              include_softmax,
                                              disable_partitioning)
                report.add(repetition, model_report)
            
            reports.append(report)

        return reports

    def _evaluate(self,
                  model,
                  gaussian_cal_set,
                  evaluation_set,
                  partition_algs,
                  metrics,
                  include_softmax=False,
                  disable_partitioning=True):
        self.model_report = ModelReport(model)

        print_start(model.name)
        print_progress("Calculate reliability scores of {model}".format(model=model.name))

        rel_analyzer = self.create_rel_analyzer_for(model)

        result = self._evaluate_model(model, evaluation_set, rel_analyzer, include_softmax)

        self._evaluate_and_report_metrics(self.load_std_metrics(), result)

        predictions = model.predict(gaussian_cal_set.X)
        features = model.project(gaussian_cal_set.X)
        gaussian_mixture = self.estimate_gaussian(features, predictions)

        ls_analyzer = ReliabilitySpecificManifoldAnalyzer(model=model, 
                                                          test_data=Dataset(X=features, Y=predictions), 
                                                          rel_analyzer=rel_analyzer
        )
        success_probs = ls_analyzer.sample(gaussian_mixture)

        self.model_report.add_convergence(success_probs)

        if not disable_partitioning:
            for partition_alg in partition_algs:
                print_progress("Estimated reliability scores based on {approach}".format(approach=partition_alg.name))

                partition_map = ls_analyzer.analyze(partition_alg)
                
                estimated_rel_scores = partition_map.calc_scores(result.features)
                result = EvaluationResult(correct_idxs=result.correct_idxs,
                                        incorrect_idxs=result.incorrect_idxs,
                                        evaluation_set=result.evaluation_set,
                                        features=result.features,
                                        rel_scores=estimated_rel_scores,
                                        softmax=result.softmax
                )

                self._evaluate_and_report_metrics(metrics, result)

        print_end()

        return self.model_report

    def estimate_gaussian(self, features, predictions):
        '''Estimates the gaussian mixture for a set of features and predictions of a given dnn model.'''
        raise NotImplementedError
    
    def create_rel_analyzer_for(self, model):
        '''Creates a reliability analyzer for a given DNN model.'''
        raise NotImplementedError
    
    def load_std_metrics(self):
        return [TrueSuccessProbability(), AverageReliabilityScores()]
    
    def _evaluate_and_report_metrics(self, metrics, result):
        for metric in metrics:
            metric.apply(result)
            metric.print_result()

            self.model_report.add(metric.copy())

    def _evaluate_model(self, model, evaluation_set, rel_analyzer, include_softmax):
        predictions = model.predict(evaluation_set.X)
        features = model.project(evaluation_set.X)
        rel_scores = calc_rel_scores(features, rel_analyzer)
        softmax = model.softmax(evaluation_set.X) if include_softmax else None
        diffs = predictions - evaluation_set.Y
        return EvaluationResult(correct_idxs=diffs==0,
                                    incorrect_idxs=diffs!=0, 
                                    evaluation_set=evaluation_set,
                                    features=features, 
                                    rel_scores=rel_scores,
                                    softmax=softmax
        )

def calc_rel_scores(features, rel_analyzer):
        result = rel_analyzer.analyze(Dataset(X=features, Y=np.zeros((len(features)))))
        return result.reliability_scores