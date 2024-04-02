from dnn.dataset import Dataset
from latentspace.analyzer import ReliabilitySpecificManifoldAnalyzer
from commons.print import print_progress, print_result
from evaluation.metrics import TrueSuccessProbability, AverageReliabilityScores, PearsonCorrelation
from evaluation.visual import histoplot, scatterplot
from commons.ops import find_type

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

class Evaluation:
    def evaluate(self, 
                 models, 
                 gaussian_cal_set, 
                 evaluation_set, 
                 rel_analyzer,
                 partition_algs, 
                 metrics,
                 include_softmax=False):
        for model in models:
            print_progress("Calculate reliability scores")

            # Model must be updated because we iterate over a set of models
            rel_analyzer.model = model

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

            evaluate_metrics(self._load_std_metrics(), result)

            histoplot(scores_correct=result.get_correct_scores(), 
                      scores_incorrect=result.get_incorrect_scores(), 
                      title="Calculated reliability score distribution",
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

                histoplot(scores_correct=result.get_correct_scores(), 
                              scores_incorrect=result.get_incorrect_scores(), 
                              title="Reliability score distribution based on {approach}".format(approach=partition_alg.name),
                              show_plot=True)

                evaluate_metrics(metrics, result)

    def estimate_gaussian(self, features, predictions):
        '''Estimates the gaussian mixture for a set of features and predictions of a given dnn model.'''
        raise NotImplementedError
    
    def _load_std_metrics(self):
        return [TrueSuccessProbability(), AverageReliabilityScores()]
    

def calc_rel_scores(features, rel_analyzer):
        result = rel_analyzer.analyze_feature_space(Dataset(X=features, Y=np.zeros((len(features)))))
        return result.reliability_scores

def evaluate_metrics(metrics, result):
    for metric in metrics:
        metric.apply(result)
        metric.print_result()
        if metric.is_visualizable():
            metric.visualize()