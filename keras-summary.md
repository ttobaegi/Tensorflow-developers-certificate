**Keras** 
  - easy-to-use deep learning library for Theano and TensorFlow 
  - provides a high-level neural networks API to develop and evaluate deep learning models.

</br>

> ### Table of Contents
> * [Basic Example](#BasicExample)
> * [TensorFlow Data Services](#data)
> * [Preprocessing](#preprocessing)
> * [Layers](#layers)
> * [Models](#models)
> * [Activation Functions](#activations)
> * [Optimizers](#optimizers)
> * [Loss Functions](#loss)
> * [Hyperparameters](#parameters)
> * [Metrics](#metrics)
> * [Visualizations](#viz)
> * [Callbacks](#callbacks)
> * [Transfer Learning](#transfer)
> * [Overfitting](#overfit)
> * [Examples](#examples)

</br>


<a name="BasicExample"></a>

## Basic Example
```py
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

data = np.random.random((1000,100))
labels = np.random.randit(2, size=(1000,1))

model = Sequential()
model.add(Dense(32,
                activation = 'relu',
                input_dim = 100))
model.add(Dense(1,
                activation = 'sigmoid'))
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.fit(data,
          labels,
          epochs = 10,
          batch_size = ['accuracy'])
predictions = model.predict(data)                
```
</br>
</br>


<a name="data"></a>

## TensorFlow Data 

**TensorFlow Datasets** 
- a collection of datasets ready to use, with TensorFlow or other Python ML frameworks enabling easy-to-use and high-performance input pipelines.
  - [`tf.data.Datasets`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
    - [guide](https://www.tensorflow.org/datasets/overview)
    - [list of datasets](https://www.tensorflow.org/datasets/catalog).


```py
# tensorflow dataset
from keras.datasets import button_housing, mnist, cifar10, imdb

# load dataset
(x_train, y_train),(x_test,y_test) = mnist.load_data()
(x_train2, y_train2),(x_test2,y_test2) = boston_housing.load_data()
(x_train3, y_train3),(x_test3,y_test3) = cifar10.load_data()
(x_train4, y_train4),(x_test4,y_test4) = imdb.load_data(num_words = 20000)

# number of classes for classification tasks > output units
num_classes = 10 
```

</br>
</br>

<a name="preprocessing"></a>

## Preprocessing
### Sequence Padding
#### Tokenizer, Text-to-sequence & Padding

```py
## Import modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

## Dataset
sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]


## Tokenizer
tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
# Key value pair (word: token)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

## Text-to-sequence
# Lists of tokenized sentences
sequences = tokenizer.texts_to_sequences(sentences)

## Padding
# Padded tokenized sentences
padded = pad_sequences(sequences, maxlen=5)

print("\nWord Index = " , word_index)
print("\nSequences = " , sequences)
print("\nPadded Sequences:")
print(padded)

```


### Other
```py
from urllib.request import urlopen
from numpy as np

data = np.loadtxt(urlopen('url...'), 
                  delimiter = ",")
# numpy slicing
X = data[:,0:9]
```
</br>
</br>


<a name="preprocessing"></a>

## Preprocessing
### Sequence Padding
#### Tokenizer, Text-to-sequence & Padding

```py
## Import modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

## Dataset
sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

## Tokenizer
tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
# Key value pair (word: token)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

## Text-to-sequence
# Lists of tokenized sentences
sequences = tokenizer.texts_to_sequences(sentences)

## Padding
# Padded tokenized sentences
padded = pad_sequences(sequences, maxlen=5)

print("\nWord Index = " , word_index)
print("\nSequences = " , sequences)
print("\nPadded Sequences:")
print(padded)

```

### One-hot Encoding

```python
from keras.utils import to_categorical

```


### ImageDataGenerator

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image augmentation
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/tmp/horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        '/tmp/validation-horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')
```

### Train and Test set split
```py
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.33, 
                                                    random_state=42)
```

### Standardization/Normalization
```py
from sklearn.preprocessing import StandardScaler

# 표준화 
scaler = StandardScaler()

```
