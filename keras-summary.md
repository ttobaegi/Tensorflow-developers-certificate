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
