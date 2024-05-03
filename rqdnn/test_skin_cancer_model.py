from evaluation.skin_cancer import SkinCancerModel
from PIL import Image
import numpy as np
import keras
from evaluation.skin_cancer import SkinCancerEvaluation



if __name__ == '__main__':
    model = SkinCancerModel(100)

    # Create Test-Data
    x1 = Image.open("../test.jpg")
    x1 = np.array(x1)
    x1 = np.expand_dims(x1, axis=0)
    x2 = x1
    X = np.concatenate((x1, x2), axis=0)

    # Test softmax
    softmax = model.softmax(X)
    print("SOFTMAX")
    print(softmax)

    # Test predict
    prediction = model.predict(x1)
    print("PREDICTION")
    print(prediction)

    # Test predict_all
    predictions = model.predict_all(X)
    print("PREDICTIONS")
    print(predictions)

    # Test Get Confidences
    confidences = model.get_confidences(x1)
    print("GET-CONFIDENCES")
    print(confidences)

    # Test Confidence
    confidence = model.confidence(x1, 5)
    print("CONFIDENCE CLASS 5")
    print(confidence)

    # Test Project
    project = model.project(x1)
    print("PROJECT")
    print(project)

    # Test Project all
    project_all = model.project_all(X)
    print("PROJECT ALL")
    print(project_all)

    # Test Get Confidences for Feature
    #feature = np.zeros((1, 6144))
    feature = np.zeros((1, 128))
    get_confidences_for_feature = model.get_confidences_for_feature(feature)
    print("CONFIDENCE FEATURE")
    print(get_confidences_for_feature)

    # Test Evaluation
    print("START")
    eval = SkinCancerEvaluation()
    eval.evaluate()
