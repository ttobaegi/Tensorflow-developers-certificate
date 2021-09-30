# Tensorflow-developers-certificate


## Convolutional Neural Networks in TensorFlow 

>#### Table of Contents
> - [1 - Exploring a Larger Dataset](#1)
> - [2 - Augmentation: A technique to avoid overfitting](#2)
>   - [2-1. ImageDataGenerator](#2-1)
> - [3 - Transfer Learning](#3)
>   - [3-1. Download pre-trained weights](#3-1)
>   - [3-2. Load pre-trained weights](#3-2)
>   - [3-3. Output layer of pre-trained model ](#3-3)
>   - [3-4. Build the new model using pre-trained model](#3-4)
>   - [3-5. Load Image Dataset](#3-5)
>   - [3-6. Data Augmentation](#3-6)
>   - [3-7. Train the model](#3-7)
> - [4 - Multi-class Classifications](#4)
>   - [4-1. Download and unzip Image Dataset](#4-1)
>   - [4-2. Image Data Augmentation](#4-2)
>   - [4-3. Build a classifier](#4-3)

</br>
</br>

<a name='1'></a>
## 1 - Exploring a Larger Dataset


</br>
</br>

<a name='2'></a>
## 2 - Augmentation 
- A technique to avoid overfitting

 작은 이미지 데이터 셋으로 NN모델을 훈련하면 과적합 overfitting 문제로 새로운 이미지 데이터에 대해 좋은 성능을 내지 못할 수 있다. 간단한 **이미지 전처리 기법 Image Augmentation으로 과적합 문제를 해결**할 수 있다.

 **"Image Augmentation"은 이미지를 훈련할 때마다 이미지에 임의로 변형을 가함으로써 모델이 더 많은 이미지를 학습하도록 만드는 것**이다. 모델이 작은 이미지 데이터 셋에 최대한 많은 정보를 뽑아내서 학습할 수 있게 한다. 

- [ImageDataGenerator](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), [작은 데이터셋으로 강력한 이미지 분류 모델 설계하기
](https://keraskorea.github.io/posts/2018-10-24-little_data_powerful_model/)

<a name='2-1'></a>
### ImageDataGenerator


- `ImageDataGenerator()`
  모델 학습 중 이미지에 임의변형 & 정규화 적용
  
     |Arguments | Meaning |
   |:---:|-|  
   | **rotation_range**| 이미지 회전 범위 (0~180)|
   | **width_shift, height_shift** |수평/수직으로 랜덤평행 이동 범위 (원본 가로, 세로 길이에 대한 비율 값)
   | **rescale** | 0-255 RGB > 1/255로 스케일링 0-1 범위로 변환 (제일 먼저 수행)|
   | **shear_range**| 전단 변환(shearing transformation) 이미지를 어긋나보이게하는 범위|
   | **zoom_range**| 랜덤 확대/축소 범위|
   | **horizontal_flip**| True : 50% 확률로 이미지를 수평으로 뒤집는다.</br>원본에 수평 비대칭성이 없을 때 (뒤집어도 자연스러운 경우) 효과적이다.|
   | **fill_mode**| 이미지 회전/이동/축소시 빈 공간을 채우는 방식 {‘constant’, ‘nearest’, ‘reflect’, ‘wrap’}|
   
 - 변형된 이미지를 배치 단위로 불러올 수 있는 generator 생성을 위한 2가지 함수
    - `.flow(data, labels)`
    - `.flow_from_directory(directory)` : to generate batches of image data (and their labels) directly from jpgs in their respective folders.
  

  
```py
# ImageDataGenerator 조건 설정 후 객체 생성
train_datagen = ImageDataGenerator(
                # value by which we will multiply the data before any other processing
                rescale=1./255,  
                
                # (0-180) range within which to randomly rotate pictures
                rotation_range=40,
                
                # ranges within which to randomly translate pictures vertically or horizontally
                width_shift_range=0.2,
                height_shift_range=0.2,
                
                # randomly applying shearing transformations
                shear_range=0.2,
                
                # randomly zooming inside pictures
                zoom_range=0.2,
                
                # randomly flipping half of the images horizontally
                horizontal_flip=True,
                
                # strategy for filling in newly created pixels (after a rotation or a width/height shift)
                fill_mode='nearest'
                                    )

train_generator = train_datagen.flow_from_directory( 
                  TRAINING_DIR,           # target directory
                  batch_size=100,        
                  class_mode='binary',    # binary classification
                  target_size=(150, 150)  # resize image to 150x150
                  )

```



</br>
</br>



<a name='3'></a>
## 3 - Transfer Learning
- [Transfer Learning](https://keras.io/guides/transfer_learning/)
- [tensorflow.keras.applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications) 
다양한 신경망 구조의 사전 훈련된 가중치를 포함하는 모듈

 

    
</br>

<a name='3-1'></a>
### 1. Download pre-trained weights

-  Transfer learning with Inceptionv3
[Inception v3](https://arxiv.org/pdf/1512.00567.pdf) 모델의 가중치를 불러와 전이학습을 수행해보자.
   - [tf.keras.applications.inception_v3.InceptionV3()](https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/InceptionV3)
   - InceptionV3 모델 구조![](https://images.velog.io/images/findingflow/post/b7e5c46b-aaa2-4d95-822a-afe566bfffcc/image.png) 

```py
import os
from tensorflow.keras import layers
from tensorflow.keras import Model

# 사전 훈련된 가중치 다운로드 : InceptionV3 
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
```
![](https://images.velog.io/images/findingflow/post/3d1f0579-357c-4c91-80c6-c99d35c55aba/image.png) 코랩에서 실행하면 tmp 폴더에 다운로드 된 것을 확인할 수 있다.

</br>

<a name='3-2'></a>
### 2. Load pre-trained weights
```py
# 사전 훈련된 가중치 로드 : InceptionV3 
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(# include_top = False인 경우 지정 필요
                                input_shape = (150, 150, 3),
                                # whether to include the fully-connected layer (last layer) at the top
                                include_top = False,
                                # None : random initialization OR imagenet (pre-training on ImageNet default)
                                weights = None)
pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers :
  # layer의 가중치 훈련 가능 여부 
  layer.trainable = False

# 모델 구조
pre_trained_model.summary()


Model: "inception_v3"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 150, 150, 3) 0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 74, 74, 32)   864         input_1[0][0]                    
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 74, 74, 32)   96          conv2d[0][0]                     
__________________________________________________________________________________________________
activation (Activation)         (None, 74, 74, 32)   0           batch_normalization[0][0]        
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 72, 72, 32)   9216        activation[0][0]                 
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 72, 72, 32)   96          conv2d_1[0][0]                   
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 72, 72, 32)   0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 72, 72, 64)   18432       activation_1[0][0]               
...
...
__________________________________________________________________________________________________
activation_68 (Activation)      (None, 7, 7, 192)    0           batch_normalization_68[0][0]     
__________________________________________________________________________________________________
activation_69 (Activation)      (None, 7, 7, 192)    0           batch_normalization_69[0][0]     
__________________________________________________________________________________________________
**mixed7 (Concatenate)            (None, 7, 7, 768)    0           activation_60[0][0]              
                                                                 activation_63[0][0]              
                                                                 activation_68[0][0]              
                                                                 activation_69[0][0]              
...
```
</br>

<a name='3-3'></a>
### 3. pre-trained model 마지막 레이어 설정, 출력 확인
`'mixed7'`이라는 레이어를 사전 훈련된 신경망 모델의 마지막 레이어로 설정하고 출력값을 확인한다.
```py 
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# 출력값 shape 확인
last layer output shape:  (None, 7, 7, 768)
```
</br>

<a name='3-4'></a>
### 4. Build the new model using pre-trained model

- `layers.Flatten()` : 3에서 설정한 마지막 레이어의 출력을 펼친다.
- `layers.Dense()` : 2개의 완전 연결 레이어 사이에 
- `layers.Dropout()` : 드롭아웃 레이어를 추가한다.
  - 이전 레이어의 출력 유닛이 무작위로 제거한다.
```py
## pre-trained model 위에 쌓을 레이어들 : x
# Flatten the output layer(last_layer) to 1D
x = layers.Flatten()(last_output)
# Fully Connected layer with 1024 hidden units & ReLU activation
x = layers.Dense(1024, activation = 'relu')(x)
# Dropout rate = 0.2 이전 레이어의 출력 유닛의 20%를 임의로 제거
x = layers.Dropout(0.2)(x)
# Final sigmoid layer for classification
x = layers.Dense(1, activation = 'sigmoid')(x)
```
- `Model()` : pre-trained model의 입력과 새롭게 구성한 레이어를 입력해 새로운 모델을 만든다.
```py
from tensorflow.keras import Model
## pre-trained model의 입력과 x 로 새로운 모델 구성
model = Model(pre_trained_model.input, x)

# 최적화 알고리즘
from tensorflow.keras.optimizers import RMSprop

## 모델 컴파일
model.compile(optimizer = RMSprop(learning_rate = 0.0001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
```

</br>

<a name='3-5'></a>
### 5. Load Image Dataset


```py
!gdown --id 1RL0T7Rg4XqQNRCkjfnLo4goOJQ7XZro9

import os
import zipfile

zip_ref = zipfile.ZipFile("./cats_and_dogs_filtered.zip", 'r')
zip_ref.extractall("tmp/")
zip_ref.close()
```



</br>

<a name='3-6'></a>
### 6. Data Augmentation

- **directory 생성 & 지정**
```py
# Define example directories and files
base_dir = 'tmp/cats_and_dogs_filtered'
train_dir = os.path.join( base_dir, 'train')
validation_dir = os.path.join( base_dir, 'validation')

## training
# Directory with training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats') 
# Directory with training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs') 

## validation
# Directory with validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats') 
# Directory with validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)
```
</br>

- **훈련 이미지 데이터 전처리 (Augmentation)**
```py
# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
                                train_dir,
                                batch_size = 20,
                                class_mode = 'binary', 
                                target_size = (150, 150)
                                )   
```
</br>

- **검증 이미지 데이터 전처리 (Augmentation)**
validation 검증 이미지 데이터에는 정규화(rescale)만 수행한다. 
```py
# validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory( 
                                     validation_dir,
                                     batch_size  = 20,
                                     class_mode  = 'binary', 
                                     target_size = (150, 150)
                                     )
```
</br>

<a name='3-7'></a>
### 7. Train the model
- [model.fit](https://keras.io/api/models/model_training_apis/)
  - `steps_per_epochs` : 한 번의 epoch에서 훈련에 사용할 batch 개수
  - `validation_steps` : 한 번의 epoch가 끝날 때, 테스트에 사용되는 batch 개수

```py
# train the model & training result
history = model.fit(
                     # input training data
                     train_generator,
                     # test data
                     validation_data = validation_generator,
                     # Integer or None 
                     steps_per_epoch = 100,
                     epochs = 20,
                     validation_steps = 50,
                     verbose = 2
                   )
Epoch 1/20
100/100 - 165s - loss: 0.3474 - accuracy: 0.8590 - val_loss: 0.1221 - val_accuracy: 0.9490
Epoch 2/20
100/100 - 157s - loss: 0.2164 - accuracy: 0.9150 - val_loss: 0.1193 - val_accuracy: 0.9580
...
Epoch 18/20
100/100 - 155s - loss: 0.1374 - accuracy: 0.9540 - val_loss: 0.1142 - val_accuracy: 0.9680
Epoch 19/20
100/100 - 155s - loss: 0.1361 - accuracy: 0.9585 - val_loss: 0.1497 - val_accuracy: 0.9660
Epoch 20/20
100/100 - 156s - loss: 0.1441 - accuracy: 0.9540 - val_loss: 0.1076 - val_accuracy: 0.9700
```

```py
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()


plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```
![](https://images.velog.io/images/findingflow/post/02652c54-f454-42df-8e86-3c0a412d8a8b/image.png)![](https://images.velog.io/images/findingflow/post/9ef54430-247c-4f1d-a327-5b45dab8ca72/image.png)



</br>
</br>

<a name='4'></a>
## 4 - Multiclass Classifications

이미지를 확인하고 가위/바위/보 세 개의 클래스를 분류하는 모델을 구현한다.

<a name='4-1'></a>
### 1. Download and unzip Image Dataset

- 훈련 데이터 셋을 다운로드 & 압축 풀기

```py
## Download dataset
# rps training set
!gdown --id 1DYVMuV2I_fA6A3er-mgTavrzKuxwpvKV
# rps testing set
!gdown --id 1RaodrRK1K03J_dGiLu8raeUynwmIbUaM

# Unzip dataset
import os
import zipfile

local_zip = './rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/rps-train')
zip_ref.close()

local_zip = './rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/rps-test')
zip_ref.close()
```
![](https://images.velog.io/images/findingflow/post/d588c087-205f-43ac-9372-2de4f702da27/image.png)

- 경로 지정
훈련에 사용될 이미지 경로를 클래스별로 지정하고, 
```py
base_dir = 'tmp/rps-train/rps'

rock_dir = os.path.join(base_dir, 'rock')
paper_dir = os.path.join(base_dir, 'paper')
scissors_dir = os.path.join(base_dir, 'scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])
```



- 이미지 확인
클래스별 이미지를 확인한다.

```py
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 2

next_rock = [os.path.join(rock_dir, fname) 
                for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname) 
                for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname) 
                for fname in scissors_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_rock+next_paper+next_scissors):
  #print(img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()
```



</br>

<a name='4-2'></a>
### 2. Image Data Augmentation

- 훈련데이터
```py
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

# Data Augmentation for Training Data
TRAINING_DIR = 'tmp/rps-train/rps'
training_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = .2,
    height_shift_range = .2,
    shear_range = .2,
    zoom_range = .2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size = (150,150),
    class_mode = 'categorical',
    batch_size = 126
)
```
Found 2520 images belonging to 3 classes.

- 검증 데이터 
```py
# Rescaling & Resizing Validation Data
VALIDATION_DIR = 'tmp/rps-test/rps-test-set'
validation_datagen = ImageDataGenerator(rescale = 1./255)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)
```
Found 372 images belonging to 3 classes.



</br>

<a name='4-3'></a>
### 3. Build a classifier


```py
model = tf.keras.models.Sequential([
        # 1st Conv layer
        tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        # 2nd Conv layer
        tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # 3rd Conv layer
        tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # 4th layer
        tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # Flatten & Dropout
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(.5),
        # 2 FC layers
        tf.keras.layers.Dense(512, activation = 'relu'),
        tf.keras.layers.Dense(3, activation = 'softmax')  # multi-class classification
])

```

</br>


- **Model Architecture**
```py
model.summary()

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_94 (Conv2D)           (None, 148, 148, 64)      1792      
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 74, 74, 64)        0         
_________________________________________________________________
conv2d_95 (Conv2D)           (None, 72, 72, 64)        36928     
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 36, 36, 64)        0         
_________________________________________________________________
conv2d_96 (Conv2D)           (None, 34, 34, 128)       73856     
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 17, 17, 128)       0         
_________________________________________________________________
conv2d_97 (Conv2D)           (None, 15, 15, 128)       147584    
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 7, 7, 128)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 6272)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 6272)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 512)               3211776   
_________________________________________________________________
dense_3 (Dense)              (None, 3)                 1539      
=================================================================
Total params: 3,473,475
Trainable params: 3,473,475
Non-trainable params: 0
_________________________________________________________________
```





</br>
</br>
