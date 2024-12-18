from evaluation.mnist import MNISTEvaluation
from evaluation.cifar10 import CIFAR10Evaluation
from evaluation.fashion import FashionMNISTEvaluation
from evaluation.skin_cancer import SkinCancerEvaluation
from evaluation.report import EvaluationReport

if __name__ == '__main__':
    eval = MNISTEvaluation()
    eval.train_models()
    mnist_reports = eval.evaluate()

    eval = CIFAR10Evaluation()
    eval.train_models()
    cifar_reports = eval.evaluate()

    eval = FashionMNISTEvaluation()
    eval.train_models()
    fashion_report = eval.evaluate()

    #eval = SkinCancerEvaluation()
    #eval.train_models()
    #skin_report = eval.evaluate()