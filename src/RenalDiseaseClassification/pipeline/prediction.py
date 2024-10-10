import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import load_img, img_to_array


import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        # load model
        model_path = r"E:\Education\Data Science\MLOPS\Project1_Renal_Disease_Classification\artifacts\training\model.h5"
        self.model = load_model(model_path)

        imagename = self.filename
        test_image = load_img(imagename, target_size = (224,224))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(self.model.predict(test_image), axis=1)
        print(result)

        if result[0] == 1:
            prediction = 'Tumor'
            return [{ "image" : prediction}]
        else:
            prediction = 'Normal'
            return [{ "image" : prediction}]