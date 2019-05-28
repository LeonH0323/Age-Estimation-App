# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:30:12 2019

@author: Lee
"""

from keras.applications import ResNet50, InceptionResNetV2
from keras.layers import Dense
from keras.models import Model
from keras.models import load_model
from keras import backend as K
import os
import cv2
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path
from keras.utils.data_utils import get_file

def get_model(model_name="ResNet50"):
    
    base_model = None
    
    if model_name == "ResNet50":
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3), pooling="avg")
    elif model_name == "InceptionResNetV2":
        base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299,299,3), pooling="avg")
        
    prediction = Dense(units=101, kernel_initializer="he_normal", use_bias=False, 
                       activation="softmax",name="pred_age")(base_model.output)
    
    model = Model(inputs=base_model.input, outputs=prediction)
    
    return model

def main():
#    root="./checkpoints"
#    weight_file="InceptionResNetV2weights.019-3.100-4.255.hdf5"
    imageName = "face.jpg"
    img_size = 224
    model = load_model('model_weight_3.133-4.723.hdf5')
    model.summary()
    image = np.empty((1, img_size, img_size, 3))
    image[0] = cv2.resize(cv2.imread(str(imageName),1),(img_size,img_size))
    result = model.predict(image)
#    print(result)
    ages_out = np.arange(0, 101).reshape(101, 1)
    predicted_ages = result.dot(ages_out).flatten()
    
    print(predicted_ages[0])
    
if __name__ == '__main__':
    main()
