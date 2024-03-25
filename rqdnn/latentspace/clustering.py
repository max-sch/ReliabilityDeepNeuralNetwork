from sklearn.mixture import GaussianMixture

import numpy as np

def estimate_init_means(features, predictions, num_labels):
    num_feature_dims = len(features[0])
    means_init = np.zeros((num_labels, num_feature_dims))
    for i in range(num_labels):
        label_idxs = np.arange(len(features))[np.array(predictions) == i]
        equal_labeled_features = np.array(features)[label_idxs,:]
        num_labels = len(equal_labeled_features)
        acc_features = np.matmul(np.transpose(equal_labeled_features), np.ones((num_labels, 1))).reshape((num_feature_dims))
        means_init[i,:] = acc_features / num_labels

    return means_init

class GaussianClusterAnalyzer:
    def __init__(self, means_init) -> None:
        self.gaussian_mixture = GaussianMixture(n_components=means_init.shape[0], means_init=means_init) 

    def estimate(self, features):
        self.gaussian_mixture = self.gaussian_mixture.fit(features)

    def sample_features(self, num_samples=1000):
        return self.gaussian_mixture.sample(num_samples)[0]      