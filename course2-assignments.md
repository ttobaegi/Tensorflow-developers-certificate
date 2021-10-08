## COURSE 3 - Convolutional Neural Networks in TensorFlow
>### Table of Contents   
> - [WEEK 1 Exploring a Larger Dataset](#3-1)        
> - [WEEK 2 Augmentation: A technique to avoid overfitting](#3-2)         
> - [WEEK 3 Transfer Learning](#3-3)          
> - [WEEK 4 Multiclass Classifications](#3-4)          

</br>
</br>

<a name='3-1'></a>
### WEEK 1. Exploring a Larger Dataset
- `flow_from_directory` on the ImageGenerator
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
- `Validation accuracy` 
  - a better indicator of model performance with new images than training accuracy.
  - based on images that the model hasn't been trained with.
- `Overfitting` 
  - more likely to occur on smaller datasets. (less likelihood of all possible features being encountered in the training process.)


</br>
</br>

<a name='3-2'></a>
### WEEK 2. Augmentation: A technique to avoid overfitting
- `Image Augmentation`
  - Manipulates the **training set** to generate more scenarios for features in the images **to solve overfitting**.
  - When using Image Augmentation, my training gets little **slower**. Because the image processing takes cycles.
  - effectively simulates having a larger data set for training.
  - When using Image Augmentation with the ImageDataGenerator, **Nothing happens to your raw image data on-disk, all augmentation is done in-memory**
   - `ImageDataGenerator()`
     - `fill_mode` : attempts to recreate lost information after a transformation like a shear. (이미지 회전/이동/축소시 빈 공간을 채우는 방식 {‘constant’, ‘nearest’, ‘reflect’, ‘wrap’})
     - `horizontal_flip` : If my training data only has people facing left, but I want to classify people facing right, how would I avoid overfitting?

</br>
</br>

<a name='3-3'></a>
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

<a name='3-4'></a>
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
