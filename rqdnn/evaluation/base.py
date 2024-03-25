from dnn.dataset import Dataset
from latentspace.analyzer import ReliabilitySpecificManifoldAnalyzer
from commons.print import print_progress, print_result
from evaluation.metrics import TrueSuccessProbability, AverageReliabilityScores
from evaluation.visual import histoplot

import numpy as np

class EvaluationResult:
    def __init__(self, 
                 diffs, 
                 features_correct, 
                 features_incorrect, 
                 scores_correct, 
                 scores_incorrect) -> None:
        self.diffs = diffs
        self.features_correct = features_correct
        self.features_incorrect = features_incorrect
        self.scores_correct = scores_correct
        self.scores_incorrect = scores_incorrect

class Evaluation:
    def evaluate(self, 
                 models, 
                 gaussian_cal_set, 
                 evaluation_set, 
                 rel_analyzer,
                 partition_algs, 
                 metrics):
        for model in models:
            print_progress("Calculate reliability scores")

            # Model must be updated because we iterate over a set of models
            rel_analyzer.model = model

            predictions = model.predict_all(evaluation_set.X)
            diffs = predictions - evaluation_set.Y
            features_correct, features_incorrect = self._prepare_features(diffs, evaluation_set, model)
            scores_correct, scores_incorrect = self._prepare_rel_scores(features_correct, features_incorrect, rel_analyzer)
            result = EvaluationResult(diffs=diffs, 
                                      features_correct=features_correct, 
                                      features_incorrect=features_incorrect, 
                                      scores_correct=scores_correct, 
                                      scores_incorrect=scores_incorrect
            )


            true_success = TrueSuccessProbability()
            true_prob = true_success.apply(result)
            avg_score = AverageReliabilityScores()
            avg_scores = avg_score.apply(result)

            print_result(metric=true_success.name, values=true_prob)
            print_result(metric=avg_score.name, values=avg_scores)

            histoplot(scores_correct, scores_incorrect, title="Calculated reliability score distribution")

            predictions = model.predict_all(gaussian_cal_set.X)
            features = model.project_all(gaussian_cal_set.X)

            gaussian_mixture = self.estimate_gaussian(features, predictions)

            ls_analyzer = ReliabilitySpecificManifoldAnalyzer(model=model, 
                                                              test_data=Dataset(X=features, Y=predictions), 
                                                              rel_analyzer=rel_analyzer
            )

            for partition_alg in partition_algs:
                print_progress("Estimated reliability scores based on {approach}".format(approach=partition_alg.name))

                partition_map = ls_analyzer.analyze(gaussian_mixture, partition_alg)

                for metric in metrics:
                    scores_correct = partition_map.calc_scores(features_correct)
                    scores_incorrect = partition_map.calc_scores(features_incorrect)
                    result = EvaluationResult(diffs=diffs, 
                                              features_correct=features_correct, 
                                              features_incorrect=features_incorrect, 
                                              scores_correct=scores_correct, 
                                              scores_incorrect=scores_incorrect
                    )

                    quantity = metric.apply(result)

                    print_result(metric=metric.name, values=quantity)

                    histoplot(scores_correct, scores_incorrect, title="Reliability score distribution based on {approach}".format(approach=partition_alg.name))

    def estimate_gaussian(self, features, predictions):
        '''Estimates the gaussian mixture for a set of features and predictions of a given dnn model.'''
        raise NotImplementedError
    
    def _prepare_features(self, diffs, evaluation_set, model):
        incorrect_idxs = diffs != 0
        X_incorrect = evaluation_set.X[incorrect_idxs,:]
        features_incorrect = model.project_all(X_incorrect)

        correct_idxs = diffs == 0
        X_correct = evaluation_set.X[correct_idxs,:]
        features_correct = model.project_all(X_correct)

        return features_correct, features_incorrect
    
    def _prepare_rel_scores(self, features_correct, features_incorrect, rel_analyzer):
        scores_correct = rel_analyzer.analyze_feature_space(Dataset(X=features_correct, Y=np.zeros((len(features_correct)))))
        scores_correct = list(zip(*scores_correct.reliability_scores))[1]

        scores_incorrect = rel_analyzer.analyze_feature_space(Dataset(X=features_incorrect, Y=np.zeros((len(features_incorrect)))))
        scores_incorrect = list(zip(*scores_incorrect.reliability_scores))[1]

        return scores_correct, scores_incorrect
