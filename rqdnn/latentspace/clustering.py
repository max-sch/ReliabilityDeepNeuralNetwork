from sklearn.mixture import GaussianMixture
from commons.print import print_info

import numpy as np

def estimate_init_means(features, predictions, num_labels):
    num_feature_dims = len(features[0])
    means_init = []
    for i in range(num_labels):
        label_idxs = np.arange(len(features))[np.array(predictions) == i]
        equal_labeled_features = np.array(features)[label_idxs,:]

        num_labels = len(equal_labeled_features)
        if num_labels == 0:
            print_info("There are no predicted labels for class {}".format(str(i)))
            continue

        acc_features = np.matmul(np.transpose(equal_labeled_features), np.ones((num_labels, 1))).reshape((num_feature_dims))
        means_init.append(acc_features / num_labels)

    return np.array(means_init)

class GaussianClusterAnalyzer:
    def __init__(self, means_init) -> None:
        self.gaussian_mixture = GaussianMixture(n_components=means_init.shape[0], means_init=means_init) 

    def estimate(self, features):
        self.gaussian_mixture = self.gaussian_mixture.fit(features)

    def sample_features(self, num_samples=1000):
        return self.gaussian_mixture.sample(num_samples)[0]      