# In[20]: RODAR COM CNN+CoreML36

from __future__ import print_function


import coremltools
from keras.models import load_model
from keras import utils
model = load_model('MeliponasImageDatastore227x227val_loss 00462 - val_acc 09718.h5')
output_labels = ['fasciculata','flavolineata', 'fuliginosa', 'melanoventer', 'nebulosa', 'paraensis', 'puncticolis', 'seminigra']
# For the first argument, use the filename of the newest .h5 file in the notebook folder.
coreml_mnist = coremltools.converters.keras.convert(
    model, input_names=['image'], output_names=['output'],
    class_labels=output_labels, image_input_names='image')

print(coreml_mnist)

coreml_mnist.author = 'Ana Carolina'
coreml_mnist.license = 'Siravenha'
coreml_mnist.short_description = 'Image based bee recognition (DigitalBees)'
coreml_mnist.input_description['image'] = 'Beewing image'
coreml_mnist.output_description['output'] = 'Probability of each specie'
coreml_mnist.output_description['classLabel'] = 'Labels of species'
from random import randint
coreml_mnist.save(f'MeliponasImageDatastore227x227val_loss 00462 - val_acc 09718.mlmodel')
