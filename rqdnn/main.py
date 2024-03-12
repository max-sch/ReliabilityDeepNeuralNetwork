from dnn.dataset import MNISTDataset, Dataset
from dnn.model import MNISTTestModel
from latentspace.analyzer import ReliabilitySpecificManifoldAnalyzer
from latentspace.clustering import GaussianClusterAnalyzer

import numpy as np

if __name__ == '__main__':
    num_features = 100
    test_data = MNISTDataset.create_randomly(num_features)
    model = MNISTTestModel()

    predictions = model.predict_all(test_data.X)
    features = model.project_all(test_data.X)

    num_labels = 10
    num_feature_dims = len(features[0])
    means_init = np.zeros((num_labels, num_feature_dims))
    for i in range(num_labels):
        label_idxs = np.arange(num_features)[np.array(predictions) == i]
        equal_labeled_features = np.array(features)[label_idxs,:]
        num_labels = len(equal_labeled_features)
        acc_features = np.matmul(np.transpose(equal_labeled_features), np.ones((num_labels, 1))).reshape((num_feature_dims))
        means_init[i,:] = acc_features / num_labels

    cluster_analyzer = GaussianClusterAnalyzer(means_init)
    cluster_analyzer.estimate(features)

    partitionMap = ReliabilitySpecificManifoldAnalyzer(
        model=model,
        test_data=Dataset(X=features, Y=predictions)
    ).analyze_gaussian(cluster_analyzer)

    print("Done")
    