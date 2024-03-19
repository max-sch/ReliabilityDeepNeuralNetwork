from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

import numpy as np

class GaussianClusterAnalyzer:
    def __init__(self, means_init) -> None:
        self.gaussian_mixture = GaussianMixture(n_components=means_init.shape[0], means_init=means_init) 

    def estimate(self, features):
        self.gaussian_mixture = self.gaussian_mixture.fit(features)

    def sample_features(self, num_samples=1000):
        return self.gaussian_mixture.sample(num_samples)[0]      