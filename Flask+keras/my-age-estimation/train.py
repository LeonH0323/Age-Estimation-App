# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:34:49 2019

@author: Lee
"""
from keras.applications import ResNet50, InceptionResNetV2, VGG16
from keras.layers import Dense,GlobalAveragePooling2D,Flatten,Dropout
from keras.models import Model
from keras import backend as K
import random
import math
from pathlib import Path
from PIL import Image
import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD, Adam
import pandas as pd
import cv2
from keras.utils import Sequence, to_categorical
import Augmentor
import matplotlib.pyplot as plt



def get_augmentor():
    
    p = Augmentor.Pipeline()
    p.flip_left_right(probability=0.5)
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    p.zoom_random(probability=0.5, percentage_area=0.95)
    p.random_distortion(probability=1, grid_width=2, grid_height=2, magnitude=8)
    p.random_color(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_contrast(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_brightness(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_erasing(probability=0.5, rectangle_area=0.2)
    
    def augmentor_image(image):
        
        image = [Image.fromarray(image)]
        for operation in p.operations:
            r = round(random.uniform(0,1), 1)
            if r <= operation.probability:
                image = operation.perform_operation(image)
        return image[0]
    
    return augmentor_image

class FanceGenerator(Sequence):
    
    def __init__(self, appa_dir, batch_size=32, image_size=224):
        
        self.image_path_and_age= []
        self._load_appa(appa_dir)
#        data = np.load("train_path_age.npz")
#        self.image_path_and_age=data['train_path_age']
#        np.savez("train_path_age",train_path_age=self.image_path_and_age)
#              
#        self.image_path_and_age= []
#         self._load_appa(appa_dir)
#        data = np.load("train_path_age.npz")      
#        self.image_path_and_age=data['image_path_and_age']
        
        self.image_num = len(self.image_path_and_age)
        self.batch_size = batch_size
        self.image_size = image_size
        self.indices = np.random.permutation(self.image_num)
        self.augmentor_image = get_augmentor()
    
    
    def __len__(self):
        
        return self.image_num // self.batch_size
    
    def __getitem__(self,idx):
        
        batch_size = self.batch_size
        image_size = self.image_size
        
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size,1), dtype=np.int32)
        
        sample_indices = self.indices[idx * batch_size:(idx + 1) * batch_size]
        
        for i, sample_id in enumerate(sample_indices):
            
            image_path, age = self.image_path_and_age[sample_id]
            image = cv2.imread(str(image_path))
            if image is None:
                print(image_path)
            x[i] = self.augmentor_image(cv2.resize(image,(image_size,image_size)))
            age = int(age)
            age += math.floor(np.random.randn() * 2 + 0.5)
            y[i] = np.clip(age, 0, 101)
        
        return x, to_categorical(y,101)
    
    def on_epoch_end(self):
        
        self.indices = np.random.permutation(self.image_num)
        
    def _load_appa(self, appa_dir):
        
        appa_root = Path(appa_dir)
        train_image_dir = appa_root.joinpath("train")
        gt_train_path = appa_root.joinpath("gt_avg_train_my.csv")
        df = pd.read_csv(str(gt_train_path))
        
        for i,row in df.iterrows():
            age = min(100,int(row.apparent_age_avg))
            image_path = train_image_dir.joinpath(row.file_name + "_face.jpg")
            print(i)
            if image_path.is_file():
                self.image_path_and_age.append([str(image_path), age])
            
        

class ValGenerator(Sequence):
    
    def __init__(self, appa_dir, batch_size=32, image_size=224):
        
        self.image_path_and_age = []
        self._load_appa(appa_dir)
#        np.savez("val_path_age",val_path_age=self.image_path_and_age)
#        self.image_path_and_age = np.load("val_path_age.npz")
        
#        self.image_path_and_age = []
#         self._load_appa(appa_dir)
#        data = np.load("val_path_age.npz")
#        self.image_path_and_age = data['val_path_age']
        
        self.image_num = len(self.image_path_and_age)
        self.batch_size = batch_size
        self.image_size = image_size
        
    def __len__(self):
        
        return self.image_num // self.batch_size
    
    def __getitem__(self, idx):
        
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, 1), dtype=np.int32)
        
        for i in range(batch_size):
            image_path, age = self.image_path_and_age[idx * batch_size + i]
            image = cv2.imread(str(image_path))
            x[i] = cv2.resize(image, (image_size, image_size))
            y[i] = age
            
        
        return x, to_categorical(y,101)
    
    def _load_appa(self, appa_dir):
        
        appa_root = Path(appa_dir)
        val_image_dir = appa_root.joinpath("valid")
        gt_val_path = appa_root.joinpath("gt_avg_valid_my.csv")
        df = pd.read_csv(str(gt_val_path))
        
        for i, row in df.iterrows():
            
            age = min(100, int(row.apparent_age_avg))
            image_path = val_image_dir.joinpath(row.file_name + "_face.jpg")
            print(i)
            if image_path.is_file():
                self.image_path_and_age.append([str(image_path), age])


        

def age_mae(y_true, y_pred):
    
    true_age = K.sum(y_true * K.arange(0, 101, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, 101, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae


def get_model(model_name="ResNet50"):
    
    base_model = None
    
    if model_name == "ResNet50":
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3), pooling="avg")
    elif model_name == "InceptionResNetV2":
        base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299,299,3), pooling="avg")
    elif model_name == "VGG16":
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))
#         for layer in base_model.layers:
#           layer.trainable = True
        
#         x = GlobalAveragePooling2D()(base_model.output)
        x = Flatten(name='flatten')(base_model.output)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dropout(0.5)(x)
        prediction = Dense(units=101, kernel_initializer="he_normal", use_bias=False, activation="softmax",name="pred_age")(x)
        model = Model(inputs=base_model.input, outputs=prediction)
        return model
    
    prediction = Dense(units=101, kernel_initializer="he_normal", use_bias=False, 
                       activation="softmax",name="pred_age")(base_model.output)
    
    model = Model(inputs=base_model.input, outputs=prediction)
    
    return model

class Schedule:
    
    def __init__(self, nb_epochs, initial_lr):
        
        self.epochs = nb_epochs
        self.initial_lr = initial_lr
        
    def __call__(self, epoch_idx):
        
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.5:
            return self.initial_lr * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.04
        return self.initial_lr * 0.008


def get_optimizer(opt_name, lr):
    
    if opt_name == "sgd":
        return SGD(lr=lr, momentum=0.9, nesterov=True)
    elif opt_name =="adam":
        return Adam(lr=lr)
    else:
        raise ValueError("optimizer name should be 'sgd' or 'adam'")

def main():
    
    appa_dir='./appa-real/appa-real-release'
    output_dir_name = 'checkpoints'
    model_name = 'VGG16'
    batch_size = 32
    epochs = 2
    lr = 0.001
    opt_method = get_optimizer('adam', lr)
    
    
    if model_name == "ResNet50":
        
        image_size = 224
    elif model_name == "InceptionResNetV2":
        
        image_size = 299
    elif model_name == "VGG16":
    
        image_size = 224
    
    train_gen = FanceGenerator(appa_dir,batch_size,image_size)
    val_gen = ValGenerator(appa_dir,batch_size,image_size)
    model = get_model(model_name=model_name)
#    
    model.compile(optimizer=opt_method, loss="categorical_crossentropy", metrics=[age_mae])
    model.summary()
##    
    output_dir='./'+output_dir_name
    output_dir = Path(__file__).resolve().parent.joinpath(output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    callbacks = [LearningRateScheduler(schedule=Schedule(epochs,lr)),
                ModelCheckpoint(str(output_dir) + "/weights.{epoch:03d}-{val_loss:.3f}-{val_age_mae:.3f}.hdf5",
                                monitor="val_age_mae",
                                verbose=1,
                                save_best_only=True,
                                mode="min")
            ]
    
    hist = model.fit_generator(generator=train_gen,
                               epochs=epochs,
                               validation_data=val_gen,
                               verbose=1,
                               callbacks=callbacks)
    
    
    print(hist.history.keys())
    print(hist.history.keys())
    plt.plot(hist.history['age_mae'])
    plt.plot(hist.history['val_age_mae'])
    plt.title('model ame')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train','test'],loc='upper left') 
    plt.show()
    
    np.savez(str(output_dir + "/history.npz"), history=hist.history)
    
    


if __name__ == '__main__':
    
    main()