# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.models import Sequential


# Librariesauxiliares
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def append_ext(fn):
    return fn+".jpg"
traindf=pd.read_csv("./dataset/tag_treinamento.csv",dtype=str)
testdf=pd.read_csv("./dataset/tag_teste.csv",dtype=str)
traindf["id"]=traindf["id"].apply(append_ext)
testdf["id"]=testdf["id"].apply(append_ext)
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

train_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="./dataset/treinamento/",
x_col="id",
y_col="label",
subset="training",
batch_size=60,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(60,60))

valid_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="./dataset/treinamento/",
x_col="id",
y_col="label",
subset="validation",
batch_size=60,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(60,60))

test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
dataframe=testdf,
directory="./dataset/teste/",
x_col="id",
y_col=None,
batch_size=60,
seed=42,
shuffle=False,
class_mode=None,
target_size=(60,60))

#build the model
print('build the model')
model = Sequential()
model.add(Conv2D(60, (3, 3), padding='same',
                 input_shape=(60,60,3)))
model.add(Activation('relu'))
model.add(Conv2D(60, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(120, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(120, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))
model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])

#fitting the model
print('fitting the model')

STEP_SIZE_TRAIN=train_generator.n#//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n #valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n #//test_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
)

#evaluete the model
print('evaluete the model')
model.evaluate_generator(generator=valid_generator,
steps=STEP_SIZE_TEST)

#predict 
print('predict')
test_generator.reset()
pred=model.predict_generator(test_generator,
steps=2,#STEP_SIZE_TEST,
verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)
print('FIM')