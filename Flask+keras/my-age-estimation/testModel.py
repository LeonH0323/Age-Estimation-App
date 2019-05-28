# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 17:00:07 2019

@author: Lee
"""
from keras.applications import ResNet50, InceptionResNetV2
from keras.layers import Dense
from keras.models import Model
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
    root="./checkpoints"
    weight_file="weights.028-3.133-4.723.hdf5"
    img_size = 224
    batch_size = 32
    model = get_model()
    model.load_weights(Path(root).joinpath(weight_file))
    appa_dir='./appa-real/appa-real-release'
    test_image_dir = Path(appa_dir).joinpath('test')
    gt_test_path = Path(appa_dir).joinpath("gt_avg_test.csv")
    image_paths = list(test_image_dir.glob("*_face.jpg"))
    
    faces = np.empty((batch_size, img_size, img_size, 3))
    ages = []
    image_names = []
    
    for i, image_path in tqdm(enumerate(image_paths)):
        faces[i % batch_size] = cv2.resize(cv2.imread(str(image_path), 1), (img_size, img_size))
        image_names.append(image_path.name[:-9])

        if (i + 1) % batch_size == 0 or i == len(image_paths) - 1:
            results = model.predict(faces)
            ages_out = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results.dot(ages_out).flatten()
            ages += list(predicted_ages)
    
    name2age = {image_names[i]: ages[i] for i in range(len(image_names))}
    df = pd.read_csv(str(gt_test_path))
    appa_abs_error = 0.0
    real_abs_error = 0.0

    for i, row in df.iterrows():
        appa_abs_error += abs(name2age[row.file_name] - row.apparent_age_avg)
        real_abs_error += abs(name2age[row.file_name] - row.real_age)

    print("MAE Apparent: {}".format(appa_abs_error / len(image_names)))
    print("MAE Real: {}".format(real_abs_error / len(image_names)))

if __name__ == '__main__':
    main()
