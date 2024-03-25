from dnn.dataset import MNISTDataset, Dataset
from dnn.model import MNISTTestModel, MNISTTestModel2, MNISTTestModel3
from latentspace.analyzer import ReliabilitySpecificManifoldAnalyzer
from latentspace.clustering import GaussianClusterAnalyzer
from evaluation.mnist import MNISTEvaluation

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    MNISTEvaluation().evaluate()
    