

## COURSE 1. Introduction to TensorFlow

</br>

### WEEK 1


- `!pip install tensorflow==2.5.0`

#### Quiz 1
- The diagram for traditional programming had Rules and Data In, `Answers` came out?
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
```

```
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

