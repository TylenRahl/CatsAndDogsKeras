import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
%matplotlib inline

#Only for GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPU's Avaliable: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#importing files not folders
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

from google.colab import drive

drive.mount('/content/gdrive')

#organise data into test,train split
os.chdir('/content/gdrive/MyDrive/CatsvsDog/train')
if os.path.isdir('train/dog') is False:
  os.makedirs('train/dog')
  os.makedirs('train/cat')
  os.makedirs('valid/dog')
  os.makedirs('valid/cat')
  os.makedirs('test/dog')
  os.makedirs('test/cat')

  for c in random.sample(glob.glob('cat*'), 500):
      shutil.move(c, 'train/cat')
  for c in random.sample(glob.glob('dog*'), 500):
      shutil.move(c, 'train/dog')
  for c in random.sample(glob.glob('cat*'), 100):
      shutil.move(c, 'valid/cat')
  for c in random.sample(glob.glob('dog*'), 100):
      shutil.move(c, 'valid/dog')
  for c in random.sample(glob.glob('cat*'), 50):
      shutil.move(c, 'test/cat')
  for c in random.sample(glob.glob('dog*'), 50):
      shutil.move(c, 'test/dog')

os.chdir('../../')

train_path = '/content/gdrive/MyDrive/CatsvsDog/train/train'
valid_path = '/content/gdrive/MyDrive/CatsvsDog/train/valid'
test_path = '/content/gdrive/MyDrive/CatsvsDog/train/test'


train_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size = (224,224), classes = ['cat','dog'], batch_size = 10)


valid_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input)\
.flow_from_directory(directory=valid_path, target_size = (224,224), classes = ['cat','dog'], batch_size = 10)

test_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input)\
.flow_from_directory(directory=test_path, target_size = (224,224), classes = ['cat','dog'], batch_size = 10, shuffle = False)

assert train_batches.n == 1000
assert valid_batches.n == 200
assert test_batches.n == 100
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 2

imgs, labels = next(train_batches)

#This function will plot images in the form of a grid with 1 row with 10 colummns where images are placed

def plotImages(images_arr):
  fig, axes = plt.subplots(1,10, figsize =(20,20))
  axes = axes.flatten()
  for img, ax  in zip( images_arr, axes):
    ax.imshow(img)
    ax.axis('off')
  plt.tight_layout
  plt.show

  plotImages(imgs)
  print(labels)

model = Sequential([
      Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
      MaxPool2D(pool_size=(2, 2), strides=2),
      Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
      MaxPool2D(pool_size=(2, 2), strides=2),
      Flatten(),
      Dense(units=2, activation='softmax'),
  ])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy',
                metrics=['accuracy'])  # for sigmoid  chose sigmoid and verbose =1

model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)

# predict
test_imgs, test_labels = next(test_batches)
plotImages(test_imgs)
print(test_labels)

test_batches.classes

predictions = model.predict(x = test_batches, batch_size = 10, verbose = 0)

np.round(predictions)

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis = -1))


def plot_confusion_matrix(cm, classes,
                          normalise=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    '''Prints and plots confusion matrix'''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalised confusion matrix")

    else:
        print("confusion matrix, without normalisation")

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[-1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("predicted label")

test_batches.class_indices

cm_plot_labels = ['cat', 'dog']
plot_confusion_matrix(cm=cm, classes = cm_plot_labels, title = 'confusion matrix')

vgg16_model = tf.keras.applications.vgg16.VGG16()

vgg16_model.summary()

# to insure the model has downloaded correctly
def count_params(model):
  non_trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.non_trainable_weights])
  trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_weights])
  return {'non_trainable_params': non_trainable_params, 'trainable_params': trainable_params}

# to insure the model has downloaded correctly
params = count_params(vgg16_model)
assert params['non_trainable_params'] == 0
assert params['trainable_params'] == 138357544

type(vgg16_model)

# removing the final LAYER
model = Sequential()
for layer in vgg16_model.layers[:-1]:
  model.add(layer)

model.summary()

# to insure the model has downloaded correctly
params = count_params(vgg16_model)
assert params['non_trainable_params'] == 0
assert params['trainable_params'] == 138357544

#FREEZES LAYERS
for layer in model.layers:
  layer.trainable = False

# changing the final output so it's only 2
model.add(
    Dense(units=2, activation='softmax'))  # If you keep pressing this it will keep adding more to the model lolz

model.summary()

params = count_params(model)
assert params['non_trainable_params'] == 134260544
assert params['trainable_params'] == 8194
# No commas 134,268,738

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics = ['accuracy'])

model.fit(x=train_batches, validation_data=valid_batches, epochs=1, verbose=2)

assert model.history.history.get('accuracy')[-1] >0.95

predictions = model.predict(x = test_batches, verbose = 0)

test_batches.classes

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

test_batches.class_indices

cm_plot_labels = ['cat', 'dog']
plot_confusion_matrix(cm=cm, classes = cm_plot_labels, title = 'confusion matrix')