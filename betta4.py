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
model.add(MaxPooling2D(pool_size=(2, 2)))# TensorFlow e tf.keras
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.models import Sequential


# Librariesauxiliares
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def append_ext(fn):
    return fn+".jpg"
traindf=pd.read_csv("D:/UFRJ - Pós Graduação/Nomografia/dataset/tag_treinamento.csv",dtype=str)
testdf=pd.read_csv("D:/UFRJ - Pós Graduação/Nomografia/dataset/tag_teste.csv",dtype=str)
traindf["id"]=traindf["id"].apply(append_ext)
testdf["id"]=testdf["id"].apply(append_ext)
datagen=ImageDataGenerator(rescale=1./255.
                           ,validation_split=0.25
                           , horizontal_flip=True
                           , rotation_range=45)
# =============================================================================
# datagen=ImageDataGenerator(rescale=1./255.
#                            ,validation_split=0.25
#                            , horizontal_flip=True
#                            , rotation_range=45
#                            ,featurewise_center=True
#                            ,featurewise_std_normalization=True                           
#                            ,width_shift_range=.15
#                            ,height_shift_range=.15                     
#                            ,zoom_range=0.5)                           
# 
# =============================================================================
train_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="D:/UFRJ - Pós Graduação/Nomografia/dataset/treinamento/",
x_col="id",
y_col="label",
subset="training",
batch_size=45,
shuffle=True,
class_mode="categorical",
target_size=(60,60))

valid_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="D:/UFRJ - Pós Graduação/Nomografia/dataset/treinamento/",
x_col="id",
y_col="label",
subset="validation",
batch_size=45,
shuffle=True,
class_mode="categorical",
target_size=(60,60))

test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
dataframe=testdf,
directory="D:/UFRJ - Pós Graduação/Nomografia/dataset/teste/",
x_col="id",
y_col="label",
shuffle=False,
class_mode=None,
batch_size=1,
target_size=(60,60))

model = Sequential()
model.add(Conv2D(15, (3, 3), strides=(1, 1), padding='same',input_shape=(60,60,3),activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(30, (3, 3), strides=(2, 2), padding='same',activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(60, (2, 2), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(120, (2, 2), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(4096,activation='relu'))
model.add(Dense(6, activation='sigmoid'))
model.summary()

epochs=15
# ruim 30 acertos 
#model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
#bom ... 44
model.compile(optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999),loss="categorical_crossentropy",metrics=["accuracy"])
#bom....43
#model.compile(optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999),loss="categorical_crossentropy",metrics=["accuracy"])
#bom....43
#model.compile(optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999),loss="categorical_crossentropy",metrics=["accuracy"])


#fitting the model
print('fitting the model')

STEP_SIZE_TRAIN=train_generator.n#//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n #valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n #//test_generator.batch_size

STEP_SIZE_TRAIN=train_generator.n#//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n #valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n #//test_generator.batch_size

gerado = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=epochs
)

#evaluate the model
print('evaluate the model')
evaluate = model.evaluate_generator(generator=valid_generator,
steps=STEP_SIZE_TEST)

#predict 
print('predict')
test_generator.reset()
pred=model.predict_generator(test_generator, #test_generator,
steps=STEP_SIZE_TEST #2
,verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
print(predicted_class_indices)
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)


import matplotlib.pyplot as plt
plt.plot(gerado.history["acc"])
plt.plot(gerado.history['val_acc'])
plt.plot(gerado.history['loss'])
plt.plot(gerado.history['val_loss'])
plt.title("Modelo de Acurácia")
plt.ylabel("Acurácia")
plt.xlabel("Epocas")
plt.legend(["Acurácia","Acurácia da Validação","Perda","Perda de Validação"])
plt.show()
print('FIM')


print(traindf)
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
