from dnn.dataset import MNISTDataset, Dataset
from dnn.model import MNISTTestModel, MNISTTestModel2, MNISTTestModel3
from latentspace.analyzer import ReliabilitySpecificManifoldAnalyzer
from latentspace.clustering import GaussianClusterAnalyzer

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    #train_data = MNISTDataset.create_less_train()
    #test_data = MNISTDataset.create_test()

    #model = MNISTTestModel3(model_file=None)
    #model.train_and_save_model(train_data=train_data, test_data=test_data)

    test_data = MNISTDataset.create_first()
    #test_data = MNISTDataset.create_randomly(1000)
    model = MNISTTestModel3()

    predictions = model.predict_all(test_data.X)
    features = model.project_all(test_data.X)

    num_labels = 10
    num_feature_dims = len(features[0])
    means_init = np.zeros((num_labels, num_feature_dims))
    for i in range(num_labels):
        label_idxs = np.arange(test_data.size())[np.array(predictions) == i]
        equal_labeled_features = np.array(features)[label_idxs,:]
        num_labels = len(equal_labeled_features)
        acc_features = np.matmul(np.transpose(equal_labeled_features), np.ones((num_labels, 1))).reshape((num_feature_dims))
        means_init[i,:] = acc_features / num_labels

    cluster_analyzer = GaussianClusterAnalyzer(means_init)
    cluster_analyzer.estimate(features)

    analyzer = ReliabilitySpecificManifoldAnalyzer(
        model=model,
        test_data=Dataset(X=features, Y=predictions)
    )
    partitionMap, reliability_scores = analyzer.analyze_gaussian2(cluster_analyzer)
    
    test_data = MNISTDataset.create_second()
    #test_data = MNISTDataset.create_randomly(1000)
    calc_rel_scores = np.zeros((test_data.size()))
    predictions = model.predict_all(test_data.X)
    diffs = predictions - test_data.Y

    print("True success probability: {true_prob}".format(true_prob=(len(diffs) - np.count_nonzero(diffs)) / len(diffs)))
    
    # Calculate rel scores and visualize with good and bad samples
    incorrect_idxs = diffs != 0
    X_incorrect = test_data.X[incorrect_idxs,:]
    features_incorrect = model.project_all(X_incorrect)
    scores_incorrect = analyzer.rel_analyzer.analyze_feature_space(Dataset(X=features_incorrect, Y=np.zeros((len(features_incorrect))))).reliability_scores
    scores_incorrect = list(zip(*scores_incorrect))[1]
    n1 = len(scores_incorrect)
    calc_rel_scores[range(n1)] = scores_incorrect
    types = ["Malicious"] * n1

    correct_idxs = diffs == 0
    X_correct = test_data.X[correct_idxs,:]
    features_correct = model.project_all(X_correct)
    scores_correct = analyzer.rel_analyzer.analyze_feature_space(Dataset(X=features_correct, Y=np.zeros((len(features_correct))))).reliability_scores
    scores_correct = list(zip(*scores_correct))[1]
    n2 = len(scores_correct)
    calc_rel_scores[range(n1,n1+n2)] = scores_correct
    types = types + ["Good"] * n2

    levels = np.unique(calc_rel_scores).tolist()
    ordered_levels = sorted(levels, reverse=True)
    ordered_levels = [str(v) for v in ordered_levels]

    rel_scores = [str(v) for v in calc_rel_scores]
    d={"Success levels":rel_scores, "Types":types}
    dataframe=pd.DataFrame(data=d)
    dataframe['Success levels'] = pd.Categorical(dataframe['Success levels'], ordered_levels)
    sns.histplot(dataframe, x="Success levels", hue="Types", discrete=True).set(title="Calculated scores")
    #plt.xlabel('Categories')
    #plt.ylabel('Frequency')
    #plt.title('Histogram of Categorical Values')
    plt.show()

    # Estimate rel scores and visualize with good and bad samples
    estimated_rel_scores = np.zeros((test_data.size()))

    estimated_scores_incorrect = partitionMap.calc_scores(features_incorrect)
    n1 = len(estimated_scores_incorrect)
    estimated_rel_scores[range(n1)] = estimated_scores_incorrect
    types = ["Malicious"] * n1

    estimated_scores_correct = partitionMap.calc_scores(features_correct)
    n2 = len(estimated_scores_correct)
    estimated_rel_scores[range(n1, n1+n2)] = estimated_scores_correct
    types = types + ["Good"] * n2

    levels = np.unique(estimated_rel_scores).tolist()
    ordered_levels = sorted(levels, reverse=True)
    ordered_levels = [str(v) for v in ordered_levels]

    rel_scores = [str(v) for v in estimated_rel_scores]
    d={"Success levels":rel_scores, "Types":types}
    dataframe=pd.DataFrame(data=d)
    dataframe['Success levels'] = pd.Categorical(dataframe['Success levels'], ordered_levels)
    sns.histplot(dataframe, x="Success levels", hue="Types", discrete=True).set(title="Estimated scores")
    plt.show()

    # Estimate rel scores and visualize with good and bad samples
    knn_rel_scores = np.zeros((test_data.size()))
    features, est_scores = list(zip(*reliability_scores))
    success_levels = np.unique(est_scores).tolist()
    success_levels = sorted(success_levels, reverse=True)
    success_levels_map = {}
    for i in range(len(success_levels)):    
        success_levels_map[i] = success_levels[i]

    knn = KNeighborsClassifier(n_neighbors=5)
    est_score_idxs = [idx for e_score in est_scores for idx,score in success_levels_map.items() if score == e_score]
    knn.fit(X=np.array(features), y=np.array(est_score_idxs))

    knn_scores_incorrect = knn.predict(features_incorrect)
    knn_scores_incorrect = [success_levels_map[idx] for idx in knn_scores_incorrect]
    n1 = len(knn_scores_incorrect)
    knn_rel_scores[range(n1)] = knn_scores_incorrect
    types = ["Malicious"] * n1

    knn_scores_correct = knn.predict(features_correct)
    knn_scores_correct = [success_levels_map[idx] for idx in knn_scores_correct]
    n2 = len(knn_scores_correct)
    knn_rel_scores[range(n1, n1+n2)] = knn_scores_correct
    types = types + ["Good"] * n2

    levels = np.unique(knn_rel_scores).tolist()
    ordered_levels = sorted(levels, reverse=True)
    ordered_levels = [str(v) for v in ordered_levels]

    rel_scores = [str(v) for v in knn_rel_scores]
    d={"Success levels":rel_scores, "Types":types}
    dataframe=pd.DataFrame(data=d)
    dataframe['Success levels'] = pd.Categorical(dataframe['Success levels'], ordered_levels)
    sns.histplot(dataframe, x="Success levels", hue="Types", discrete=True).set(title="KNN predicted scores")
    plt.show()
    
    print("Done")
    