
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import SGD
from matplotlib import pyplot as plt

img_rows, img_cols = 229,229

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    "C:/Users/pma009/Documents/python/MeliponasImageDatastore227x227_8especies/train",
    target_size=(img_rows,img_cols),
    batch_size=32)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    "C:/Users/pma009/Documents/python/MeliponasImageDatastore227x227_8especies/test",
    target_size=(img_rows,img_cols),
    batch_size=32)


model = InceptionV3(include_top=True, weights=None, classes=8)

# callbacks_list = [
#     ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)]
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

optim = SGD(lr=0.001,momentum=0.6)
# optim = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
model.summary()
history = model.fit_generator(train_generator, steps_per_epoch=50, epochs=60, validation_data=validation_generator,validation_steps=50)#,
                              # callbacks=[early_stopping])#,
                              # callbacks=callbacks_list)
acc = history.history["val_acc"]
model.save(f'InceptionV3_MeliponasImageDatastore227x227_ValAcc_{acc[-1]}.h5')
model.save_weights(f'InceptionV3_MeliponasImageDatastore227x227_weights_ValAcc_{acc[-1]}.hdf5')


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

####### ResNet50
del train_datagen, train_generator, validation_datagen, validation_generator
img_rows, img_cols = 224,224

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    "C:/Users/pma009/Documents/python/MeliponasImageDatastore227x227_8especies/train",
    target_size=(img_rows,img_cols),
    batch_size=32)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    "C:/Users/pma009/Documents/python/MeliponasImageDatastore227x227_8especies/test",
    target_size=(img_rows,img_cols),
    batch_size=32)

from keras.callbacks import EarlyStopping
from keras.applications.resnet50 import ResNet50

model = ResNet50(include_top=True, weights=None, classes=8)#(include_top=False, weights='imagenet', classes=8)

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

optim = SGD(lr=0.001,momentum=0.6)
# optim = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
model.summary()
history = model.fit_generator(train_generator, steps_per_epoch=50, epochs=60, validation_data=validation_generator,validation_steps=50)#,
                              # callbacks=[early_stopping])#,
                              # callbacks=callbacks_list)
acc = history.history["val_acc"]
pos = len(acc)-1
model.save(f'ResNet50_MeliponasImageDatastore227x227_ValAcc_{acc[-1]}.h5')
model.save_weights(f'ResNet50_MeliponasImageDatastore227x227_weights_ValAcc_{acc[-1]}.hdf5')



print(history.history.keys())

plt.figure(2)

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


#####
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet


