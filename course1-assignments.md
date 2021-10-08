## COURSE 1 - Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning
>### Table of Contents
> - [WEEK 1 A New Programming Paradigm](#1-1)
> - [WEEK 2 Introduction to Computer Vision](#1-2)
> - [WEEK 3 Enhancing Vision with Convolutional Neural Networks](#1-3)
> - [WEEK 4 Using Real-world Images](#1-4)

</br>

<a name='1-1'></a>
### WEEK 1. A New Programming Paradigm
- `!pip install tensorflow==2.5.0`
- The diagram for traditional programming had Rules and Data In, `Answers` came out.
- The diagram for Machine Learning had Answers and Data In, `Rules` came out.
- When I tell a computer what the data represents (i.e. this data is for walking, this data is for running), what is that process called? `Labelling the Data`
- `Dense` : A layer of connected neurons
 It will then use the data that it knows about, that's the set of Xs and Ys that we've already seen to measure how good or how bad its guess was. The loss function measures this and then gives the data to the optimizer which figures out the next guess.
 - `Loss function` measures how good the current ‘guess’ is.
- `optimizer` generates a new and improved guess. how good or how badly the guess was done using the data from the loss function. 
- `Convergence` : The process of getting very close to the correct answer.
As the guesses get better and better, an accuracy approaches 100 percent, the term convergence is used.
- `model.fit` trains the neural network to fit one set of values to another.

#### Exercise 1 (Housing Prices)
  In this exercise you'll try to **build a neural network that predicts the price of a house according to a simple formula.**
So, imagine if house pricing was as easy as a house costs 50k + 50k per bedroom, so that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.
How would you create a neural network that learns this relationship so that it would predict a 7 bedroom house as costing close to 400k etc.

  Hint: Your network might work better if you scale the house price down. You don't have to give the answer 400. it might be better to create something that predicts the number 4, and then your answer is in the 'hundreds of thousands' etc.

```py
import tensorflow as tf
import numpy as np
from tensorflow import keras

# GRADED FUNCTION: house_model
def house_model(y_new):
    xs = # Your Code Here#
    ys = # Your Code Here#
    model = # Your Code Here#
    model.compile(# Your Code Here#)
    model.fit(# Your Code here#)
    return model.predict(y_new)[0]
```

```py
# GRADED FUNCTION: house_model
def house_model(y_new):
    xs = np.array([1., 2., 3., 4., 5., 6.], dtype = float)
    ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)
    model = tf.keras.Sequential([
               keras.layers.Dense(units=1, 
                                  input_shape=[1])
                                ])
    model.compile(optimizer = 'sgd',
                 loss = 'mean_squared_error')
    model.fit(xs, ys, epochs=1000)
    return model.predict(y_new)[0]
    
prediction = house_model([7.0])
print(prediction)    


Epoch 1/1000
6/6 [==============================] - 4s 653ms/sample - loss: 60864.4805
Epoch 2/1000
6/6 [==============================] - 0s 1ms/sample - loss: 28302.0078
...
Epoch 999/1000
6/6 [==============================] - 0s 10ms/sample - loss: 0.1688
Epoch 1000/1000
6/6 [==============================] - 0s 234us/sample - loss: 0.1676
[400.59048]
```

</br>
</br>

<a name='1-2'></a>
### WEEK 2. Introduction to Computer Vision
#### Exercise 2 (Handwriting Recognition)
In the course you learned how to do classificaiton using Fashion MNIST, a data set containing items of clothing. There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9. Write an **MNIST classifier** that trains to 99% accuracy or above, and does it without a fixed number of epochs -- i.e. you should stop training once you reach that level of accuracy.

1. It should succeed in less than 10 epochs, so it is okay to change epochs= to 10, but nothing larger
2. When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!"
3. If you add any additional variables, make sure you use the same names as the ones used in the class

```py
import tensorflow as tf
from os import path, getcwd, chdir

# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab mnist.npz from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
path = f"{getcwd()}/../tmp2/mnist.npz"

# GRADED FUNCTION: train_mnist
def train_mnist():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    # YOUR CODE SHOULD START HERE

    # YOUR CODE SHOULD END HERE

    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
    # YOUR CODE SHOULD START HERE

    # YOUR CODE SHOULD END HERE
    model = tf.keras.models.Sequential([
        # YOUR CODE SHOULD START HERE
    
        # YOUR CODE SHOULD END HERE
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # model fitting
    history = model.fit(# YOUR CODE SHOULD START HERE
              # YOUR CODE SHOULD END HERE
    )
    # model fitting
    return history.epoch, history.history['acc'][-1]
```

```py
# GRADED FUNCTION: train_mnist
def train_mnist():

    # 사용자 정의 콜백 클래스 
    class myCallback(tf.keras.callbacks.Callback) :
        def on_epoch_end(self, epoch, logs = {}) :
            if logs.get('accuracy') is not None and logs.get('accuracy') >= 0.99 :
                print('\nReached 99% accuracy so cancelling training!')
                self.model.stop_training = True

    # 데이터 셋 로드 
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)

    # 콜백 변수 생성 - 콜백 클래스 객체 호출
    callbacks = myCallback()
    
    # 3 -layers NN
    model = tf.keras.models.Sequential([
        # YOUR CODE SHOULD START HERE
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(512, activation = tf.nn.relu),
        tf.keras.layers.Dense(10, activation = tf.nn.softmax)
        # YOUR CODE SHOULD END HERE
    ])

    # model compile
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # model fitting
    history = model.fit( x_train, y_train, # training data
                        epochs = 10,
                        callbacks = [callbacks]
    )
    
    return history.epoch, history.history['acc'][-1]

train_mnist()


Epoch 1/10
60000/60000 [==============================] - 9s 145us/sample - loss: 2.8917 - acc: 0.9067
Epoch 2/10
...
Epoch 10/10
60000/60000 [==============================] - 9s 142us/sample - loss: 0.1856 - acc: 0.9638
([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 0.96383333)
```

</br>
</br>

<a name='1-3'></a>
### WEEK 3. Enhancing Vision with Convolutional Neural Networks

</br>
</br>



<a name='1-4'></a>
### WEEK 4. Using Real-world Images

</br>
</br>
