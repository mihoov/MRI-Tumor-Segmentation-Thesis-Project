#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import nibabel
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
import pandas as pd


# In[9]:


flair_files={}
t1_files={}
t2_files={}
t1ce_files={}
seg_files={}

file_img_root="Preprocessed_BraTS_TrainObjLocalization"

#creating list of training files for DataGenerator
for root, dirs, files in os.walk(file_img_root):
    for name in files:
        indx = name.find('_')
        key = name[indx+1:indx+6]
        file_path = os.path.join(root, name)
            
        if 'flair.nii.gz' in name:
            flair_files[key] = file_path
        elif 't1.nii.gz' in name:
            t1_files[key] = file_path
        elif 't1ce.nii.gz' in name:
            t1ce_files[key] = file_path
        elif 't2.nii.gz' in name:
            t2_files[key] = file_path
        elif 'seg.nii.gz' in name:
            seg_files[key] = file_path

#check that every segmentation file (ground truth) has corresponding X data
for k in seg_files.keys():
    if k not in flair_files.keys():
        print('Not found in flair:', k)
    if k not in t1_files.keys():
        print('Not found in t1:', k)
    if k not in t1ce_files.keys():
        print('Not found in t1ce:', k)
    if k not in t2_files.keys():
        print('Not found in t2:', k)


# In[10]:


# There is a segmentation file for samle 01627 however there does not exist a t1ce file or t2 file for 01627
# We will remove 01627 from the training data

sample_list = [k for k in seg_files.keys()]
sample_list.remove('01627')
print('01627' in sample_list)


# In[11]:

class TrainDataGenerator(keras.utils.Sequence):

    def __init__(self, samples_list, y_dict, list_X_dicts, batch_size=16, dim=(160,240,140), n_channels=1, shuffle=True):
        """Constructor can be expanded,
           with batch size, dimentation etc.
        """
        self.samples_list=samples_list
        self.y_dict=y_dict
        self.list_X_dicts=list_X_dicts
        self.batch_size=batch_size
        self.dim=dim
        self.n_channels=n_channels
        self.shuffle=shuffle
        self.on_epoch_end()

    def __len__(self):
        #Take all batches in each iteration'
        return int(np.floor(len(self.samples_list) / self.batch_size))

    def __getitem__(self, index):
        'Get next batch'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # single file
        samples_list_temp = [self.samples_list[k] for k in indexes]

        # Set of X_train and y_train
        X, y = self.__data_generation(samples_list_temp)

        return X, y

    def on_epoch_end(self):
        #Updates indexes after each epoch'
        self.indexes = np.arange(len(self.samples_list))
        #shuffle data if shuffle is true
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            

    def __data_generation(self, samples_list_temp):
        #Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim), dtype=np.uint8)
        
        # Generate data
        for i, file_key in enumerate(samples_list_temp):
            # Store sample
            X[i,] = np.stack([nibabel.load(d[file_key]).get_fdata() for d in self.list_X_dicts], axis=-1).astype('float32')

            # Store target segmentation
            y[i,] = (nibabel.load(self.y_dict[file_key]).get_fdata() > 0).astype('uint8')

        return X, y

        
# 75-20-5 train-validation-test split of data 
remain_samples, test_samples, _ , _ = train_test_split(sample_list, sample_list, test_size=0.5)

train_samples, val_samples, _ , _ = train_test_split(remain_samples, remain_samples, test_size=0.21)

#build data generators

img_data_gen_args = {'batch_size':1,
                         'dim':(160,240, 240), 
                         'n_channels':4,
                         'shuffle':True}

list_X_dicts = [flair_files, t1_files, t1ce_files, t2_files]

train_data_generator = TrainDataGenerator(samples_list = train_samples,
                                          y_dict = seg_files,
                                          list_X_dicts = list_X_dicts,
                                          **img_data_gen_args)

val_data_generator = TrainDataGenerator(samples_list = val_samples,
                                          y_dict = seg_files,
                                          list_X_dicts = list_X_dicts,
                                          **img_data_gen_args)



# In[12]:



import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, concatenate, BatchNormalization, Dense, Dropout, Flatten 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import Activation, UpSampling3D, ZeroPadding3D



def conv_block(input_layer, num_filters):
    x = Conv3D(filters=num_filters, kernel_size=3, strides=1, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv3D(filters=num_filters, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
        
    return x

def encoder_block(input_layer, num_filters):
    x = conv_block(input_layer, num_filters)
    out = MaxPooling3D((2,2,2))(x)
    
    return out, x

def decoder_block(input_layer, conc_layer, num_filters):
    x = conv_block(input_layer, num_filters)
    x = UpSampling3D(size=2)(x)
    out = concatenate([conc_layer, x])
    
    return out
    
def build_classifier(input_shape):

    input_layer = Input(input_shape)
    
    c1, u1 = encoder_block(input_layer, 8)
    c2, u2 = encoder_block(c1,16)
    c3, u3 = encoder_block(c2, 32)
    c4, u4 = encoder_block(c3, 64)
    
    c5 = decoder_block(c4, u4, 64)
    c6 = decoder_block(c5, u3, 32)
    c7 = decoder_block(c6, u2, 16)
    c8 = decoder_block(c7, u1, 8)
    
    segmentation_layer = Conv3D(filters=1, kernel_size=1, activation='sigmoid', padding='same')(c8)
    
    model = Model(input_layer, segmentation_layer, name='3D_semantic_segmentation')

    return model


#free up RAM
keras.backend.clear_session()

#build model
input_shape = (160, 240, 240, 4)
model = build_classifier(input_shape)
print(model.summary())


# In[ ]:



checkpoint = keras.callbacks.ModelCheckpoint('MRI_segmentation.h5', save_weights_only=True)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=10)

callbacks = [checkpoint, early_stopping]
#learning_rate=lr_schedule
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['acc'])

epochs=20
history = model.fit(train_data_generator , 
                              validation_data=val_data_generator,
                             epochs=epochs,
                             verbose=1,
                             callbacks=callbacks)


# In[ ]:




