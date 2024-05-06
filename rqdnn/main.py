from evaluation.mnist import MNISTEvaluation
from evaluation.cifar10 import CIFAR10Evaluation
from evaluation.fashion import FashionMNISTEvaluation
from evaluation.skin_cancer import SkinCancerEvaluation
from evaluation.report import EvaluationReport

if __name__ == '__main__':
    eval = MNISTEvaluation()
    mnist_reports = eval.evaluate()

    eval = CIFAR10Evaluation()
    cifar_reports = eval.evaluate()

    eval = FashionMNISTEvaluation()
    fashion_report = eval.evaluate()

    eval = SkinCancerEvaluation()
    skin_report = eval.evaluate()

    EvaluationReport.export(mnist_reports, cifar_reports, fashion_report, skin_report)