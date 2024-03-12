from dnn.dataset import MNISTDataset
from dnn.model import MNISTTestModel
from latentspace.analyzer import ReliabilitySpecificManifoldAnalyzer

if __name__ == '__main__':
    partitionMap = ReliabilitySpecificManifoldAnalyzer(
        model=MNISTTestModel(),
        test_data=MNISTDataset.create_test()
    ).analyze()

    print("Done")
    