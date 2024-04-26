from evaluation.mnist import MNISTEvaluation
from evaluation.cifar10 import CIFAR10Evaluation

if __name__ == '__main__':
    eval = CIFAR10Evaluation()
    #eval.train_models()
    eval.evaluate()