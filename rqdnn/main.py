from dnn.dataset import MNISTDataset
from dnn.model import MNISTTestModel, MNISTTestModel2
from manifold.analyzer import ReliabilitySpecificManifoldAnalyzer

import os
import numpy as np

if __name__ == '__main__':
    
    #train_data = MNISTDataset.create_train()
    #test_data = MNISTDataset.create_test()

    #model = MNISTTestModel2(model_file=None)
    #model.train_and_save_model(train_data=train_data, test_data=test_data)

    partitionMap = ReliabilitySpecificManifoldAnalyzer(
        model=MNISTTestModel2(),
        test_data=MNISTDataset.create_test(),
        rel_measure=None
    ).analyze()

    print("Done")
    