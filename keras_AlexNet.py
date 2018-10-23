#!/usr/bin/env python
# coding: utf-8

# In[6]:


from __future__ import print_function

import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Activation, Dropout, Flatten, \
    Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD
# from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

# import numpy

# from keras import backend as K
# import keras.backend.tensorflow_backend as K

#import tensorflow as tf
# coremltools supports Keras version 2.0.6
print('keras version ', keras.__version__)


# In[7]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    "C:/Users/pma009/Documents/python/MeliponasImageDatastore227x227_8especies/train",
    target_size=(227,227),
    batch_size=32)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    "C:/Users/pma009/Documents/python/MeliponasImageDatastore227x227_8especies/test",
    target_size=(227,227),
    batch_size=32)


img_rows, img_cols = train_generator.image_shape[1], train_generator.image_shape[2]
# num_classes = train_generator.num_class

#
#
# # In[16]:
#
#
# model_m = Sequential()
# model_m.add(Conv2D(32, (5, 5), input_shape=train_generator.image_shape, activation='relu'))
# model_m.add(MaxPooling2D(pool_size=(2, 2)))
# model_m.add(Dropout(0.5))
# model_m.add(Conv2D(64, (3, 3), activation='relu'))
# model_m.add(MaxPooling2D(pool_size=(2, 2)))
# model_m.add(Dropout(0.2))
# model_m.add(Conv2D(128, (1, 1), activation='relu'))
# model_m.add(MaxPooling2D(pool_size=(2, 2)))
# model_m.add(Dropout(0.2))
# model_m.add(Flatten())
# model_m.add(Dense(128, activation='relu'))
# model_m.add(Dense(num_classes, activation='softmax'))
# # Inspect model's layers, output shapes, number of trainable parameters
# print(model_m.summary())
#
#
# # In[17]:
#
#
# model_c = Sequential()
# model_c.add(Conv2D(32, (3, 3), input_shape=train_generator.image_shape, activation='relu'))
# # Note: hwchong, elitedatascience use 32 for second Conv2D
# model_c.add(Conv2D(64, (3, 3), activation='relu'))
# model_c.add(MaxPooling2D(pool_size=(2, 2)))
# model_c.add(Dropout(0.25))
# model_c.add(Flatten())
# model_c.add(Dense(128, activation='relu'))
# model_c.add(Dropout(0.5))
# model_c.add(Dense(num_classes, activation='softmax'))
# # Inspect model's layers, output shapes, number of trainable parameters
# print(model_c.summary())
#
#
# # In[18]:


callbacks_list = [
    # keras.callbacks.ModelCheckpoint(
    #     filepath='best_model_Alexnet.{epoch:02d}-{val_loss:.2f}.h5',
    #     monitor='val_loss', save_best_only=True),
    # keras.callbacks.EarlyStopping(monitor='val_acc', patience=5)
    # ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True),
    # EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto', baseline=None, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)]

# # (4) Compile
# model_c.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model_c.fit_generator(train_generator, steps_per_epoch=100, epochs=10, validation_data=validation_generator,validation_steps=50,callbacks=callbacks_list)

# In[19]:
def alexnetModificada():
    # ALEXNET modificada
    # (3) Create a sequential model
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(227,227,3), strides=(4, 4),
                     kernel_size=(11,11), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096, input_shape=(256*256*3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Dense Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(8))
    model.add(Activation('softmax'))
    return model


model = alexnetModificada()

optim = SGD(lr=0.001,momentum=0.4)
# optim = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
model.summary()
history = model.fit_generator(train_generator, steps_per_epoch=50, epochs=60, validation_data=validation_generator,validation_steps=50,
                              callbacks=callbacks_list)
acc = history.history["val_acc"]
pos = len(acc)-1

model.save(f'MeliponasImageDatastore227x227_ValAcc_{acc[-1]}.h5')
model.save_weights(f'MeliponasImageDatastore227x227_weights_ValAcc_{acc[-1]}.hdf5')
#


# model = alexnet()
# optim = SGD(lr=0.0001)
# model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['acc'])
# model.summary()
# model.fit_generator(train_generator, steps_per_epoch=100, epochs=10, validation_data=validation_generator,validation_steps=50,callbacks=callbacks_list)



# model_m.compile(loss='categorical_crossentropy',
#                 optimizer='adam', metrics=['accuracy'])
# model_m.fit_generator(train_generator, steps_per_epoch=100, epochs=10, validation_data=validation_generator,validation_steps=50,callbacks=callbacks_list)
#
#
#
# model_c.compile(loss='categorical_crossentropy',
#                 optimizer='adam', metrics=['accuracy'])
# model_c.fit_generator(train_generator, steps_per_epoch=100, epochs=10, validation_data=validation_generator,validation_steps=50,callbacks=callbacks_list)


# # In[20]:
# import coremltools
# from keras.models import load_model
#
# output_labels = ['fasciculata','flavolineata', 'fuliginosa', 'melanoventer', 'nebulosa', 'paraensis', 'puncticolis', 'seminigra']
# # For the first argument, use the filename of the newest .h5 file in the notebook folder.
# coreml_mnist = coremltools.converters.keras.convert(
#     model, input_names=['image'], output_names=['output'],
#     class_labels=output_labels, image_input_names='image')
#
# print(coreml_mnist)
#
# coreml_mnist.author = 'Ana Carolina'
# coreml_mnist.license = 'Siravenha'
# coreml_mnist.short_description = 'Image based bee recognition (DigitalBees)'
# coreml_mnist.input_description['image'] = 'Beewing image'
# coreml_mnist.output_description['output'] = 'Probability of each specie'
# coreml_mnist.output_description['classLabel'] = 'Labels of species'
# from random import randint
# coreml_mnist.save(f'MeliponasImageDatastore227x227_8especiesClassifier_RMSprop_Acc {randint(0, 900)}.mlmodel')



print(history.history.keys())

plt.figure(1)

# summarize history for accuracy

plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

# summarize history for loss

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#
#
# # In[ ]:
#
#
#
#
