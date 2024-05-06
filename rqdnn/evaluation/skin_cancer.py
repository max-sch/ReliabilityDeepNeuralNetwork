
from evaluation.base import Evaluation, ModelLevel
from dnn.model import Model as Model2
from dnn.dataset import Dataset
import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
import math
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate,Dense, Conv2D, MaxPooling2D, Flatten,Input,Activation,add,AveragePooling2D,Dropout
from tensorflow.keras.preprocessing import image
from PIL import Image
from reliability.analyzer import ConformalPredictionBasedReliabilityAnalyzer
from evaluation.metrics import AverageReliabilityScores, AverageOutputDeviation, SoftmaxPositionToReliabilityCorrelation, PearsonCorrelation
from latentspace.partition_map import DecisionTreePartitioning, KnnPartitioning
from latentspace.clustering import GaussianClusterAnalyzer, estimate_init_means
from commons.ops import determine_deviation_softmax, random_splits

class_to_idx_mapper = lambda x:x

def determine_softmax_pos_fun(softmax, true_labels):
    return determine_deviation_softmax(softmax, true_labels, class_to_idx_mapper) 


class SkinCancerEvaluation(Evaluation):
    def __init__(self) -> None:
        self.skin_images_provider = SkinCancerDatasetProvider()

    def evaluate(self):
        models = self._load_models()
        gaussian_cal_set = self.skin_images_provider.create_gaussian_cal()
        evaluation_set = self.skin_images_provider.create_evaluation()

        metrics = [AverageReliabilityScores()]
        partitioning_algs = [DecisionTreePartitioning(), KnnPartitioning()]

        super().evaluate(models=models,
                        evaluation_set=evaluation_set,
                        gaussian_cal_set=gaussian_cal_set,
                        partition_algs=partitioning_algs,
                        metrics=metrics,
                        include_softmax=True)       


    def estimate_gaussian(self, features, predictions):
        means_init = estimate_init_means(features, predictions, num_labels=7)
        cluster_analyzer = GaussianClusterAnalyzer(means_init)
        cluster_analyzer.estimate(features)
        return cluster_analyzer
    
    def create_rel_analyzer_for(self, model):
        return ConformalPredictionBasedReliabilityAnalyzer(model=model,
                                                           calibration_set=self.skin_images_provider.create_cal(),
                                                           tuning_set=self.skin_images_provider.create_tuning(),
                                                           class_to_idx_mapper=class_to_idx_mapper)

    def _load_models(self):
        return [SkinCancerModel(100), SkinCancerModel(50), SkinCancerModel(1)]
    
    def _load_std_metrics(self):
        std_metrics = super()._load_std_metrics()
        std_metrics.append(AverageOutputDeviation(determine_deviation=determine_softmax_pos_fun))
        std_metrics.append(SoftmaxPositionToReliabilityCorrelation(determine_deviation=determine_softmax_pos_fun, num_pos=10))
        std_metrics.append(PearsonCorrelation(determine_deviation=determine_softmax_pos_fun))
        return None
    
class SkinCancerDatasetProvider:
    def __init__(self) -> None:
        (self.x_test, self.y_test) = self._load_data("/evaluationData/ISIC2018_Task3_Test_GroundTruth.csv", "/evaluationData/ISIC2018_Task3_Test_Images/ISIC2018_Task3_Test_Images/" )
        (self.x_cal_tun, self.y_cal_tun) = self._load_data("/evaluationData/HAM10000_metadata_cal_tun", "/AdditionalEvaluationData/" )
        (self.x_gaussian, self.y_gaussian) = self._load_data("/evaluationData/HAM10000_metadata_gaussian", "/AdditionalEvaluationData/" )
        number_of_cal_tun_data = self.y_cal_tun.shape[0]
        cal_tun_split = random_splits([math.floor(number_of_cal_tun_data * 0.5), math.floor(number_of_cal_tun_data * 0.5)])
        self.cal_idxs = [ i for i, x in enumerate(cal_tun_split == 0) if x]
        self.tun_idxs = [ i for i, x in enumerate(cal_tun_split == 1) if x]
    
    def create_gaussian_cal(self):
        # X, Y = self.x_test[self.gaussian_cal_idxs,:], self.y_test[self.gaussian_cal_idxs]
        return Dataset(self.x_gaussian, self.y_gaussian)
    
    def create_cal(self):
        X, Y = self.x_cal_tun[self.cal_idxs,:], self.y_cal_tun[self.cal_idxs]
        return Dataset(X,Y)
    
    def create_tuning(self):
        X, Y = self.x_cal_tun[self.tun_idxs,:], self.y_cal_tun[self.tun_idxs]
        return Dataset(X,Y)

    def create_evaluation(self):
        # X, Y = self.x_test[self.eval_idxs,:], self.y_test[self.eval_idxs]
        return Dataset(self.x_test, self.y_test)
    
    def _load_data(self, path_y, path_x):
        metadata = pd.read_csv(path_y)
        classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        X = np.zeros(shape=(metadata.shape[0], 299, 299, 3))
        Y = np.zeros(shape=(metadata.shape[0]))

        for j, entry in enumerate(metadata.itertuples(), 0):
            # Create X
            img = image.load_img(path_x + entry.image_id + ".jpg", target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            # OpenCV Import
            # x = cv2.imread(path_x + entry.image_id + ".jpg")
            # x = cv2.resize(x, dsize=(299, 299), interpolation=cv2.INTER_NEAREST)
            X[j] = x

            # Create Y
            Y[j] = classes.index(entry.dx)

        X = tf.keras.applications.inception_resnet_v2.preprocess_input(X)
        return (X, Y)


class SkinCancerModel(Model2):
    def __init__(self, model_level) -> None:
        self.name = "SkinCancerModel_{level}".format(level=model_level)
        self.model_file = "../" + self.name + ".hdf5"
        self.num_classes = 7
        self.input_shape = (299, 299, 3)

        self.model = self.load_from(self.model_file)

        model2 = self.load_from(self.model_file)
        self.feature_extractor = Model(inputs=model2.input, outputs=model2.get_layer(model2.layers[-2].name).output)


    def load_from(self, model_file):
        model = self.get_basic_model_instance()
        model.load_weights(model_file, skip_mismatch=False)
        return model

    def get_basic_model_instance(self):
        # Source of code: https://github.com/skrantidatta/Attention-based-Skin-Cancer-Classification/tree/main
        irv2 = tf.keras.applications.InceptionResNetV2(
        include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classifier_activation="softmax",
        )
        conv = irv2.layers[-28].output
        
        attention_layer,map2 = SoftAttention(aggregate=True,m=16,concat_with_x=False,ch=int(conv.shape[-1]),name='soft_attention')(conv)
        attention_layer=(MaxPooling2D(pool_size=(2, 2),padding="same")(attention_layer))
        conv=(MaxPooling2D(pool_size=(2, 2),padding="same")(conv))

        conv = concatenate([conv,attention_layer])
        conv  = Activation('relu')(conv)
        conv = Dropout(0.5)(conv)
        
        output = Flatten()(conv)
        output = Dense(128, activation='linear')(output)
        output = Dense(7, activation='softmax')(output)
        model = Model(inputs=irv2.input, outputs=output)

        return model

    # X has the dimension [number_of_elements, number_of_pixels_x_coordinate, number_of_pixels_y_coordinate, 3]
    def softmax(self, X):
        X = self._prepare_input(X)
        return self.model.predict(X)
    
    #TODO: Check (current dimension is equal to the dimension of the softmax output)
    def get_confidences_for_feature(self, feature) -> dict:
        softmax = self.model.layers[-1]
        softmax_predictions = softmax(feature)
        return {class_idx: float(probability) for class_idx, probability in enumerate(softmax_predictions[0])}

    def softmax_for_features(self, features):
        softmaxTest = self.model.layers[-1]
        features_test = np.expand_dims(features, axis=0)
        return softmaxTest(features_test)[0]

    def predict(self, x):
        softmax_predictions = self.softmax(x)
        return np.argmax(softmax_predictions, axis=1)
    
    def predict_all(self, X):
        softmax_predictions = self.softmax(X)
        return np.argmax(softmax_predictions, axis=1)
    
    def confidence(self, x, y): 
        return self.get_confidences(x)[y]
    
    def get_confidences(self, x) -> dict:
        x = self._prepare_input(x)
        return {class_idx: float(probability) for class_idx, probability in enumerate(self.softmax(x)[0])}

    def project(self, x):
        #x = self._prepare_input(x)
        #return np.array([float(element) for element in self.feature_extractor.predict(x)[0]])
        X = self._prepare_input(x)
        return self.feature_extractor.predict(X)
    
    def project_all(self, X):
        X = self._prepare_input(X)
        return self.feature_extractor.predict(X)

    def get_output_shape(self):
        return 7 
    
    def _prepare_input(self, X):
        # The required dimension of the input values is: [number_of_elements, 299, 299, 3]
        # X = tf.keras.applications.inception_resnet_v2.preprocess_input(X)
        #return np.resize(X, (X.shape[0],self.input_shape[0],self.input_shape[1], self.input_shape[2]))
        return X




    
# Source of code: https://github.com/skrantidatta/Attention-based-Skin-Cancer-Classification/tree/main
from keras import backend as K
from tensorflow.keras.layers import Layer,InputSpec
import tensorflow.keras.layers as kl
import tensorflow as tf

class SoftAttention(Layer):
    def __init__(self,ch,m,concat_with_x=False,aggregate=False,**kwargs):
        self.channels=int(ch)
        self.multiheads = m
        self.aggregate_channels = aggregate
        self.concat_input_with_scaled = concat_with_x

        
        super(SoftAttention,self).__init__(**kwargs)

    def build(self,input_shape):

        self.i_shape = input_shape

        kernel_shape_conv3d = (self.channels, 3, 3) + (1, self.multiheads) # DHWC
    
        self.out_attention_maps_shape = input_shape[0:1]+(self.multiheads,)+input_shape[1:-1]
        
        if self.aggregate_channels==False:

            self.out_features_shape = input_shape[:-1]+(input_shape[-1]+(input_shape[-1]*self.multiheads),)
        else:
            if self.concat_input_with_scaled:
                self.out_features_shape = input_shape[:-1]+(input_shape[-1]*2,)
            else:
                self.out_features_shape = input_shape
        

        self.kernel_conv3d = self.add_weight(shape=kernel_shape_conv3d,
                                        initializer='he_uniform',
                                        name='kernel_conv3d')
        self.bias_conv3d = self.add_weight(shape=(self.multiheads,),
                                      initializer='zeros',
                                      name='bias_conv3d')

        super(SoftAttention, self).build(input_shape)

    def call(self, x):

        exp_x = K.expand_dims(x,axis=-1)

        c3d = K.conv3d(exp_x,
                     kernel=self.kernel_conv3d,
                     strides=(1,1,self.i_shape[-1]), padding='same', data_format='channels_last')
        conv3d = K.bias_add(c3d,
                        self.bias_conv3d)
        conv3d = kl.Activation('relu')(conv3d)

        conv3d = K.permute_dimensions(conv3d,pattern=(0,4,1,2,3))

        
        conv3d = K.squeeze(conv3d, axis=-1)
        conv3d = K.reshape(conv3d,shape=(-1, self.multiheads ,self.i_shape[1]*self.i_shape[2]))

        softmax_alpha = K.softmax(conv3d, axis=-1) 
        softmax_alpha = kl.Reshape(target_shape=(self.multiheads, self.i_shape[1],self.i_shape[2]))(softmax_alpha)

        
        if self.aggregate_channels==False:
            exp_softmax_alpha = K.expand_dims(softmax_alpha, axis=-1)       
            exp_softmax_alpha = K.permute_dimensions(exp_softmax_alpha,pattern=(0,2,3,1,4))
   
            x_exp = K.expand_dims(x,axis=-2)
   
            u = kl.Multiply()([exp_softmax_alpha, x_exp])   
  
            u = kl.Reshape(target_shape=(self.i_shape[1],self.i_shape[2],u.shape[-1]*u.shape[-2]))(u)

        else:
            exp_softmax_alpha = K.permute_dimensions(softmax_alpha,pattern=(0,2,3,1))

            exp_softmax_alpha = K.sum(exp_softmax_alpha,axis=-1)

            exp_softmax_alpha = K.expand_dims(exp_softmax_alpha, axis=-1)

            u = kl.Multiply()([exp_softmax_alpha, x])   

        if self.concat_input_with_scaled:
            o = kl.Concatenate(axis=-1)([u,x])
        else:
            o = u
        
        return [o, softmax_alpha]

    def compute_output_shape(self, input_shape): 
        return [self.out_features_shape, self.out_attention_maps_shape]

    
    def get_config(self):
        return super(SoftAttention,self).get_config()
