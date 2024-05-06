from evaluation.mnist import MNISTEvaluation
from evaluation.cifar10 import CIFAR10Evaluation
from evaluation.report import EvaluationReport

if __name__ == '__main__':
    eval = MNISTEvaluation()
    mnist_reports = eval.evaluate()

    eval = CIFAR10Evaluation()
    cifar_reports = eval.evaluate()

    EvaluationReport.export(mnist_reports, cifar_reports)