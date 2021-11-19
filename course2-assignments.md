텐서플로우 개발자 자격증 준비 코세라 강의 코드 정리 (1) Introduction to TensorFlow

출처: https://geniewishescometrue.tistory.com/entry/Certificate-텐서플로우-자격증-정리-Introduction-to-TensorFlow-1?category=510887 [Findingflow]


## COURSE 2 - Convolutional Neural Networks in TensorFlow
>### Table of Contents   
> - [WEEK 1 Exploring a Larger Dataset](#2-1)        
> - [WEEK 2 Augmentation: A technique to avoid overfitting](#2-2)         
> - [WEEK 3 Transfer Learning](#2-3)          
> - [WEEK 4 Multiclass Classifications](#2-4)          

</br>
</br>

<a name='2-1'></a>
### WEEK 1. Exploring a Larger Dataset
- ImageGenerator
  - `flow_from_directory`
    - The ability to easily load images for training
    - The ability to pick the size of training images
    - The ability to automatically label images based on their directory name
- The size of output image after Convolution/Pooling layers
  - Input Image size : 150x150
    1. Pass a 3x3 Convolution over it : the size of resulting image is 148x148
    2. Use Pooling of size 2x2 : the size of resulting image is 75x75
- To view the history of training, create a variable ‘history’ and assign it to the return of model.fit or model.fit_generator
- `The model.layers API` : to inspect the impact of convolutions on the images.
- When exploring the graphs, the loss levelled out at about .75 after 2 epochs, but the accuracy climbed close to 1.0 after 15 epochs. What's the significance of this? : There was no point training after 2 epochs, as we overfit to the training data
- `Overfitting` 
    - more likely to occur on smaller datasets. (less likelihood of all possible features being encountered in the training process.)
  - `Validation accuracy` 
    - a better indicator of model performance with new images than training accuracy.
    - based on images that the model hasn't been trained with.

</br>

#### Exercise 1. Cats vs. Dogs 
```py
# ATTENTION: Please do not alter any of the provided code in the exercise. Only add your own code where indicated
# ATTENTION: Please do not add or remove any cells in the exercise. The grader will check specific cells based on the cell position.
# ATTENTION: Please use the provided epoch values when training.

# In this exercise you will train a CNN on the FULL Cats-v-dogs dataset
# This will require you doing a lot of data preprocessing because
# the dataset isn't split into training and validation for you
# This code block has all the required inputs
import os
import zipfile
import random
import tensorflow as tf
import shutil
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd

path_cats_and_dogs = f"{getcwd()}/../tmp2/cats-and-dogs.zip"
shutil.rmtree('/tmp')
local_zip = path_cats_and_dogs
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

print(len(os.listdir('/tmp/PetImages/Cat/')))
print(len(os.listdir('/tmp/PetImages/Dog/')))
# Expected Output:
# 1500
# 1500


# Use os.mkdir to create your directories
# You will need a directory for cats-v-dogs, and subdirectories for training
# and testing. These in turn will need subdirectories for 'cats' and 'dogs'
try:
    #YOUR CODE GOES HERE
except OSError:
    pass


# Write a python function called split_data which takes
# a SOURCE directory containing the files
# a TRAINING directory that a portion of the files will be copied to
# a TESTING directory that a portion of the files will be copie to
# a SPLIT SIZE to determine the portion
# The files should also be randomized, so that the training set is a random
# X% of the files, and the test set is the remaining files
# SO, for example, if SOURCE is PetImages/Cat, and SPLIT SIZE is .9
# Then 90% of the images in PetImages/Cat will be copied to the TRAINING dir
# and 10% of the images will be copied to the TESTING dir
# Also -- All images should be checked, and if they have a zero file length,
# they will not be copied over
#
# os.listdir(DIRECTORY) gives you a listing of the contents of that directory
# os.path.getsize(PATH) gives you the size of the file
# copyfile(source, destination) copies a file from source to destination
# random.sample(list, len(list)) shuffles a list
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
# YOUR CODE STARTS HERE
'''

'''
# YOUR CODE ENDS HERE

CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

print(len(os.listdir('/tmp/cats-v-dogs/training/cats/')))# 1350
print(len(os.listdir('/tmp/cats-v-dogs/training/dogs/')))# 1350
print(len(os.listdir('/tmp/cats-v-dogs/testing/cats/')))# 150
print(len(os.listdir('/tmp/cats-v-dogs/testing/dogs/')))# 150


# DEFINE A KERAS MODEL TO CLASSIFY CATS V DOGS
# USE AT LEAST 3 CONVOLUTION LAYERS
model = tf.keras.models.Sequential([
# YOUR CODE HERE
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])


#### NOTE 
# In the cell below you **MUST** use a batch size of 10 (`batch_size=10`) for the `train_generator` and the `validation_generator`. 
# Using a batch size greater than 10 will exceed memory limits on the Coursera platform.
TRAINING_DIR = #YOUR CODE HERE
train_datagen = #YOUR CODE HERE

# NOTE: YOU MUST USE A BATCH SIZE OF 10 (batch_size=10) FOR THE 
# TRAIN GENERATOR.
train_generator = #YOUR CODE HERE
VALIDATION_DIR = #YOUR CODE HERE
validation_datagen = #YOUR CODE HERE

# NOTE: YOU MUST USE A BACTH SIZE OF 10 (batch_size=10) FOR THE 
# VALIDATION GENERATOR.
validation_generator = #YOUR CODE HERE 

# Expected Output: Found 2700 images belonging to 2 classes. Found 300 images belonging to 2 classes.

history = model.fit_generator(train_generator,
                              epochs=2,
                              verbose=1,
                              validation_data=validation_generator)
```

```py

```

</br>
</br>

<a name='2-2'></a>
### WEEK 2. Augmentation: A technique to avoid overfitting
- `Image Augmentation`
  - Manipulates the **training set** to generate more scenarios for features in the images **to solve overfitting**.
  - When using Image Augmentation, my training gets little **slower**. Because the image processing takes cycles.
  - effectively simulates having a larger data set for training.
  - When using Image Augmentation with the ImageDataGenerator, **Nothing happens to your raw image data on-disk, all augmentation is done in-memory**
- `ImageDataGenerator()` : 모델 학습 중 이미지에 임의변형 & 정규화 적용

    |Arguments | Meaning |
   |:---|-|  
   | **rotation_range**| 이미지 회전 범위 (0~180)|
   | **width_shift, height_shift** |수평/수직으로 랜덤평행 이동 범위 (원본 가로, 세로 길이에 대한 비율 값)
   | **rescale** | 0-255 RGB > 1/255로 스케일링 0-1 범위로 변환 (제일 먼저 수행)|
   | **shear_range**| 전단 변환(shearing transformation) 이미지를 어긋나보이게하는 범위|
   | **zoom_range**| 랜덤 확대/축소 범위|
   | **horizontal_flip**| True : 50% 확률로 이미지를 수평으로 뒤집는다.</br>원본에 수평 비대칭성이 없을 때 (뒤집어도 자연스러운 경우) 효과적이다.|
   | **fill_mode**| attempts to recreate lost information after a transformation like a shear.</br>이미지 회전/이동/축소시 빈 공간을 채우는 방식 {‘constant’, ‘nearest’, ‘reflect’, ‘wrap’}|

  - `horizontal_flip` : If my training data only has people facing left, but I want to classify people facing right, how would I avoid overfitting? 
 - 변형된 이미지를 배치 단위로 불러올 수 있는 generator 생성을 위한 2가지 함수
    - `.flow(data, labels)`
    - `.flow_from_directory(directory)` : to generate batches of image data (and their labels) directly from jpgs in their respective folders.
</br>
</br>

<a name='2-3'></a>
### WEEK 3. Transfer Learning
- `Transfer Learning` Models 
  - `Transfer learning` is useful. Because I can use the features that were learned from large datasets that I may not have access to.
  - To use the image Augmentation, you are adding new layers at the bottom of the network, and you can use image augmentation when training these.
  - To change the number of classes the model using transfer learning, add the DNN at the bottom of the network and specify the output layer with the number of classes you want.
  - `layer.trainable = false` : lock or freeze a layer from retraining.
- `Dropouts`
  - Help avoid overfitting. Because neighbor neurons can have similar weights, and thus can skew the final training.  
  - `tf.keras.layers.Dropout(0.2)` : adding Dropout of 20% of neurons using TensorFlow
    - dropout parameter of 0.2  : I will lose 20% of the untrained nodes.
    - Symptom of a Dropout rate being set too high : the network would lose specialization to the effect that it would be inefficient or ineffective at learning, driving accuracy down

</br>
</br>

<a name='2-4'></a>
### WEEK 4. Multiclass Classifications
- The diagram for traditional programming had Rules and Data In, Answers came out.
- DNN for Fashion MNIST have `10 output neurons`, which mean the dataset has `10 classes`
- Convolution : A technique to extract features from an image
- The impact of applying Convolutions on top of a DNN depends on many factors. It might make your training faster or slower, and a poorly designed Convolutional layer may even be less efficient than a plain DNN!
- `rescale` method on an ImageGenerator is used to `normalize` the image. 
- Image augmentation with Transfer Learning : It's pre-trained layers that are frozen. So you can augment your images as you train the bottom layers of the DNN with them.
- `.flow_from_directory(..,class_mode='categorical')` : multiple classes for Image Augmentation.

</br>
</br>



