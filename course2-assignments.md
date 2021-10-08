# COURSE 3 - Convolutional Neural Networks in TensorFlow
>### Table of Contents   
> - [WEEK 1 Exploring a Larger Dataset](#3-1)        
> - [WEEK 2 Augmentation: A technique to avoid overfitting](#3-2)         
> - [WEEK 3 Transfer Learning](#3-3)          
> - [WEEK 4 Multiclass Classifications](#3-4)          

</br>
</br>

<a name='3'></a>
## COURSE 3. Convolutional Neural Networks in TensorFlow

</br>

<a name='3-1'></a>
### WEEK 1. Exploring a Larger Dataset


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


</br>
</br>

<a name='3-4'></a>
### WEEK 4. Multiclass Classifications


</br>
</br>
