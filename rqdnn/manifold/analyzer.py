from manifold.partition_map import ManifoldPartitionMap
from reliability.analyzer import ConformalPredictionBasedReliabilityAnalyzer
from dnn.dataset import MNISTDataset
import numpy as np

class ReliabilitySpecificManifoldAnalyzer:
    def __init__(self, model, test_data, rel_measure, step_size=0.01) -> None:
        self.model = model
        self.test_data = test_data
        self.rel_measure = rel_measure
        self.step_size = step_size

    def analyze(self) -> ManifoldPartitionMap:
        #error_rate = self.step_size
        #enum_dataset = {k:v for k, (v,_) in enumerate(self.test_data)}
        #lower_success_bounds = {}
        #self._determine_success(error_rate=error_rate, 
        #                        cal_data=MNISTDataset.create_cal(),
        #                        tuning_data=MNISTDataset.create_cal(), 
        #                        enum_dataset=enum_dataset, 
        #                        lower_success_bounds=lower_success_bounds)
        #
        
        rel_analyzer = ConformalPredictionBasedReliabilityAnalyzer(model=self.model,
                                                                   calibration_set=MNISTDataset.create_cal(),
                                                                   tuning_set=MNISTDataset.create_cal())
        lower_success_bounds = rel_analyzer.analyze(self.test_data)
        success = sum([p for _,p in lower_success_bounds.items()]) / self.test_data.size()
        print("Success probability: " + str(success))
            
        return None
        #rel_measures,features = self._calc_reliability_measures()

        #partition_map = ManifoldPartitionMap(bin_resolution=0.05)
        #partition_map.fit(features, rel_measures)

        #return partition_map
    
    def _determine_success(self, cal_data, tuning_data, error_rate, enum_dataset, lower_success_bounds):
        if (error_rate > 1) or (len(enum_dataset) == 0):
            return
        
        print("Error rate: " + str(error_rate))
        conf_predictor = RegularizedAdaptiveConformalPrediction(model=self.model,
                                                                calibration_set=cal_data,
                                                                tuning_data=tuning_data,
                                                                error_rate=error_rate)

        cached_keys = []
        for i,x in enum_dataset.items():
            pred_set = conf_predictor.calc_prediction_set(x)
            if (len(pred_set) == 1) and (i not in lower_success_bounds.keys()):
                lower_success_bounds[i] = 1 - error_rate
                cached_keys.append(i)

        new_error_rate = error_rate + self.step_size
        new_enum_dataset = {k:v for k,v in enum_dataset.items() if k not in cached_keys}
        self._determine_success(error_rate=new_error_rate, 
                                cal_data=cal_data,
                                tuning_data=tuning_data, 
                                enum_dataset=new_enum_dataset, 
                                lower_success_bounds=lower_success_bounds)

    def _calc_reliability_measures(self):
        rel_measures = []
        features = []

        for x,y in iter(self.test_data):
            rel_measures.append(self.rel_measure.conditional_success(x, y))
            features.append(self.model.project(x))
        
        return (rel_measures, features)