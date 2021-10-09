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
- New parogramming paradigm
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
- to **build a neural network that predicts the price of a house according to a simple formula.**
- house pricing was as easy as a house costs 50k + 50k per bedroom, so that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.
- create a neural network that learns this relationship so that it would predict a 7 bedroom house as costing close to 400k etc.
  - Hint : Your network might work better if you scale the house price down. You don't have to give the answer 400. it might be better to create something that predicts the number 4, and then your answer is in the 'hundreds of thousands' etc.

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

- Fashion MNIST : 70,000 of 28x28 Greyscale Fashion images used
- 10 output neurons : 10 different labels
- `Relu` : It only returns x if x is greater than zero
- Spliting data into training and test sets to test a network with previously unseen data.
- `on_epoch_end` method gets called when an epoch finishes.
- `callbacks=` parameter to set in fit function to tell it to use callbacks.

</br>

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
    '''
    Define Custom Callback class
    '''
    # YOUR CODE SHOULD END HERE

    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
    # YOUR CODE SHOULD START HERE
    '''
    Preprocess image data : Normalize
    '''
    # YOUR CODE SHOULD END HERE
    model = tf.keras.models.Sequential([
        # YOUR CODE SHOULD START HERE
        '''
        1 Flatten layer for image data
        2 Dense layers 
          - units of the last layer = # of classes 
          - activation function of the last layer = tf.nn.softmax
        '''
        # YOUR CODE SHOULD END HERE
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # model fitting
    history = model.fit(# YOUR CODE SHOULD START HERE
                        '''
                        x, y (training data)
                        epochs = # of epochs
                        callback =
                        '''
                        # YOUR CODE SHOULD END HERE
                       )
    # model fitting
    return history.epoch, history.history['acc'][-1]
```

```py
def train_mnist():

    # Custom Callback class 
    class myCallback(tf.keras.callbacks.Callback) :
        def on_epoch_end(self, epoch, logs = {}) :
            if (logs.get('accuracy') is not None and logs.get('accuracy') > 0.99) :
                print('\nReached 99% accuracy so cancelling training!')
                self.model.stop_training = True

    # Load Dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)

    # Normalization
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # Callback 변수 생성
    callbacks = myCallback()
    
    # 3-layers NN
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(512, activation = tf.nn.relu),
        tf.keras.layers.Dense(10, activation = tf.nn.softmax)
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

Epoch 1/9
60000/60000 [==============================] - 11s 180us/sample - loss: 0.1988 - acc: 0.9412
...
Epoch 9/9
60000/60000 [==============================] - 10s 172us/sample - loss: 0.0122 - acc: 0.9961
([0, 1, 2, 3, 4, 5, 6, 7, 8], 0.99611664)
```

</br>
</br>

<a name='1-3'></a>
### WEEK 3. Enhancing Vision with Convolutional Neural Networks

- `Convolution` : A technique to isolate features in images.
- `Pooling` : A technique to reduce the information in an image while maintaining features.
- After passing a `3x3 filter` over a 28x28 image, output will be `26x26`
- After `max pooling` a 26x26 image with a `2x2 filter`, the output will be `13x13`
- Applying Convolutions on top of our Deep neural network will make training
It depends on many factors. It might make your training faster or slower, and a poorly designed Convolutional layer may even be less efficient than a plain DNN!
</br>

#### Exercise 3 (Fashion MNIST with Convolutions)
- improve Fashion MNIST to 99.8% accuracy or more using only a single convolutional layer and a single MaxPooling 2D. 
- stop training once the accuracy goes above this amount. It should happen in less than 20 epochs, so it's ok to hard code the number of epochs for training, but your training must end once it hits the above metric. If it doesn't, then you'll need to redesign your layers.
  - When 99.8% accuracy has been hit, you should print out the string "Reached 99.8% accuracy so cancelling training!"

```py
import tensorflow as tf
from os import path, getcwd, chdir

# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab mnist.npz from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
path = f"{getcwd()}/../tmp2/mnist.npz"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# GRADED FUNCTION: train_mnist_conv
def train_mnist_conv():
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.

    # YOUR CODE STARTS HERE
    '''
    Custom Callback class - on_epoch_end()
    '''
    # YOUR CODE ENDS HERE

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)
    # YOUR CODE STARTS HERE
    '''
    
    '''
    # YOUR CODE ENDS HERE

    model = tf.keras.models.Sequential([
            # YOUR CODE STARTS HERE
            '''
            
            '''
            # YOUR CODE ENDS HERE
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model fitting
    history = model.fit(
        # YOUR CODE STARTS HERE

        # YOUR CODE ENDS HERE
    )
    # model fitting
    return history.epoch, history.history['acc'][-1]

_, _ = train_mnist_conv()
```

```py
def train_mnist_conv():

    # YOUR CODE STARTS HERE  
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs = {}):
            if (logs.get('acc') is not None and logs.get('acc') > 0.998):
                print('\nReached 99% accuracy so cancelling training!')
                self.model.stop_training = True    
    # YOUR CODE ENDS HERE

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)
    # YOUR CODE STARTS HERE
    training_images = training_images.reshape(60000, 28, 28, 1)
    test_images = test_images.reshape(10000, 28, 28, 1)
    training_images, test_images = training_images/255., test_images/255.
    # YOUR CODE ENDS HERE

    model = tf.keras.models.Sequential([
            # YOUR CODE STARTS HERE
            tf.keras.layers.Conv2D(64, (3,3), activation = tf.nn.relu, input_shape = (28,28,1)),    
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation = tf.nn.relu),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation = tf.nn.relu),
            tf.keras.layers.Dense(10, activation = tf.nn.softmax)
            # YOUR CODE ENDS HERE
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model fitting
    history = model.fit(
        # YOUR CODE STARTS HERE
        training_images, training_labels, 
        epochs=15,
        callbacks = [myCallback()]
        # YOUR CODE ENDS HERE
    )
    # model fitting
    return history.epoch, history.history['acc'][-1]
    
_, _ = train_mnist_conv()

Epoch 1/15
60000/60000 [==============================] - 14s 240us/sample - loss: 0.1184 - acc: 0.9635
...
Epoch 11/15
59680/60000 [============================>.] - ETA: 0s - loss: 0.0051 - acc: 0.9981
Reached 99% accuracy so cancelling training!
60000/60000 [==============================] - 14s 228us/sample - loss: 0.0051 - acc: 0.9981
```




</br>
</br>


<a name='1-4'></a>
### WEEK 4. Using Real-world Images

- Using Image Generator, label images based on the directory the image is contained in.
- `rescale` method on the Image Generator is used to normalize the image.
- `target_size` parameter on the training generator to specify the training size for the images.
- input_shape (300, 300, 3) : Every Image will be 300x300 pixels, with 3 bytes to define color
- `overfitting on your training data` : If your training data is close to 1.000 accuracy, but your validation data isn’t.
- The reason why `Convolutional Neural Networks` are better for classifying images like horses and humans
  - In these images, the features may be in different parts of the frame.
  - There’s a wide variety of horses.
  - There’s a wide variety of humans.
- After reducing the size of the images, the training results were different. We removed some convolutions to handle the smaller images
</br>

#### Exercise 4 (Handling complex images)
- happy or sad dataset which contains 80 images, 40 happy and 40 sad. 
- Create a convolutional neural network that trains to 100% accuracy on these images
- cancel the training upon hitting training accuracy of >.999
- Hint : it will work best with 3 convolutional layers.
```py
# GRADED FUNCTION: train_happy_sad_model
def train_happy_sad_model():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    DESIRED_ACCURACY = 0.999

    class myCallback(# your code):
         # Your Code

    callbacks = myCallback()
    
    # This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
    model = tf.keras.models.Sequential([
            # Your Code Here
            '''
            
            '''
                                      ])

    from tensorflow.keras.optimizers import RMSprop

    model.compile(# Your Code Here
                  '''
                  
                  '''
                  )
        

    # This code block should create an instance of an ImageDataGenerator called train_datagen 
    # And a train_generator by calling train_datagen.flow_from_directory

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = # Your Code Here
                    '''
                    
                    '''
    # Please use a target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory(# Your Code Here
                                                        
                                                        )
    # Expected output: 'Found 80 images belonging to 2 classes'

    # This code block should call model.fit_generator and train for
    # a number of epochs.
    # model fitting
    history = model.fit_generator( # Your Code Here  
                                 '''
                                 
                                 '''
                                 )
    # model fitting
    return history.history['accuracy'][-1]
```




</br>
</br>
