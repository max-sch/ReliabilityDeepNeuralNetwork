class ManifoldPartitionMap:
    def __init__(self, splitting_strategy=None) -> None:
        self.splitting_strategy = splitting_strategy

    def fit(self, rel_measures, features):
        raise NotImplementedError