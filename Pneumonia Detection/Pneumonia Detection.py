import matplotlib.pyplot as plt
import seaborn as sns
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
import cv2
import os

os.listdir("chest_xray/")
train_dir='chest_xray/train'
val_dir='chest_xray/test'
test_dir='chest_xray/val'

from keras.preprocessing.image import ImageDataGenerator

image_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    samplewise_center=True,
    samplewise_std_normalization=True
)

img_shape=(256,256)
# train generator for traing data
train = image_generator.flow_from_directory(train_dir,
                                            batch_size=8,
                                            shuffle=True,
                                            class_mode='binary',
                                            target_size=img_shape)
# validation generator for validation data
validation = image_generator.flow_from_directory(val_dir,
                                                batch_size=1,
                                                shuffle=False,
                                                class_mode='binary',
                                                target_size= img_shape)
#test generator for test data
test = image_generator.flow_from_directory(test_dir,
                                            batch_size=1,
                                            shuffle=False,
                                            class_mode='binary',
                                            target_size=img_shape)

early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,min_delta=1e-7,restore_best_weights=True)

reduce_lr=keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.2,patience=3,min_delta=1e-7)


def build_model(input_shape):
    inputs = keras.layers.Input(input_shape)
    # Block 1
    x = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='valid')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Dropout(.2)(x)

    # Block 2
    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Dropout(.2)(x)

    # Block 3
    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid')(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Dropout(.4)(x)

    # Head
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(.5)(x)
    # final layer
    output = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, output)

    return model

model=build_model((256,256,3))
model.compile(loss='binary_crossentropy',optimizer=tf.optimizers.Adam(lr=3e-5) ,metrics='binary_accuracy')
model.summary()

history = model.fit(train, batch_size=8, epochs=20, validation_data=validation, callbacks=[early_stopping, reduce_lr])