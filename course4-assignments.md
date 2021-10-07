>#### Table of Contents
> - [**COURSE 4 - Sequences, Time Series and Prediction**](#4)    
>      [WEEK 1 Sequences and Prediction](#4-1)  
>      [WEEK 2 Deep Neural Networks for Time Series](#4-2)    
>      [WEEK 3 Recurrent Neural Networks for time series](#4-3)  
>      [WEEK 4 Real-world time series data](#4-4)

</br>
</br>

<a name='4'></a>
## COURSE 4. Sequences, Time Series and Prediction

</br>

<a name='4-1'></a>
### WEEK 1 Sequences and Prediction
- Time Series
  - Univariate time series : single value at each time step.
    - ex. Hour by hour temperature
    ![image](https://user-images.githubusercontent.com/79742748/136352506-e8bbf02d-c29c-4a80-93dd-9b72caafa2cf.png)
  - Multivariate time series : multiple values at each time step.
    -  ex. Hour by hour weather 
     ![image](https://user-images.githubusercontent.com/79742748/136358745-83063bbe-b829-4b00-bf5c-2a4182662717.png)
     ![image](https://user-images.githubusercontent.com/79742748/136358885-b0c83a59-c041-4e41-999a-a6261acc4cf0.png)
- `imputation` : A projection of unknown (usually past or missing) data
  ![image](https://user-images.githubusercontent.com/79742748/136359563-ded8bbf7-6262-4b54-a4ae-a002bf09d6dc.png)
- A sound wave is a good example of time series data (True)
- `Seasonality` : A regular change in shape of the data. 
  - seen when patterns repeat at predictable intervals. 
- `Trend` : An overall direction for data regardless of direction
  - where time series have a specific direction that they're moving in. 
- `Noise` (In the context of time series) : Unpredictable changes in time series data
- `Autocorrelation` : Data that follows a predictable shape, even if the scale is different
  - Namely it correlates with a delayed copy of itself often called a lag. 
- `non-stationary` time series : One that has a disruptive event breaking trend and seasonality
  - the optimal time window that you should use for training will vary. 
  - stationary, meaning its behavior does not change over time.


</br>
</br>

<a name='4-2'></a>
### WEEK 2 Deep Neural Networks for Time Series
- `windowed dataset` : A fixed-size subset of a time series 
- `drop_remainder=true` : It ensures that all rows in the data window are the same length by cropping data
- `dataset = dataset.map(lambda window: (window[:-1], window[-1:]))` : split an n column window into n-1 columns for features and 1 column for a label
- MSE stands for `Mean Squared error`
- MAE stands for `Mean Absolute Error`
- time values are in time[], series values are in series[] to split the series into training and validation at time 1000

```py
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
```


- To inspect the learned parameters in a layer after training, assign a variable to the layer and add it to the model using that variable. Inspect its properties after training
  - Iterate through the layers dataset of the model to find the layer you want (x)
  - Run the model with unit data and inspect the output for that layer (x)
  - Decompile the model and inspect the parameter set for that layer (x)



- `lr property` : set the learning rate of the SGD optimizer? 
- Use a LearningRateScheduler object in the callbacks namespace and assign that to the callback to amend the learning rate of the optimizer on the fly, after each epoch




</br>
</br>


<a name='4-3'></a>
### WEEK 3 Recurrent Neural Networks for Time Series
- notation `X` : the input to an RNN, `Y(hat) and H` : the outputs
- `sequence to vector` if an RNN has 30 cells numbered 0 to 29 : The Y(hat) for the last cell
  - `vector`는 마지막 vector를 의미한다. 
  - `return_sequence = True`인 경우 vector가 아닌 sequence : The total Y(hat) for all cells
- `Lambda layer` : Allows you to execute arbitrary code while training
- tf.expand_dims(`axis = `) : Defines the dimension index at which you will expand the shape of the tensor 
- `Huber loss` : A new loss function was introduced in this module, named after a famous statistician. 
- The primary difference between a simple RNN and an LSTM : In addition to the H output, LSTMs have a cell state that runs across all cells 
- `tf.keras.backend.clear_session()`  to clear out all temporary variables that tensorflow might have from previous sessions.
- The model will fail because you need `return_sequences=True` after the first LSTM layer
```py
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
tf.keras.layers.Dense(1),
```

</br>
</br>


<a name='4-4'></a>
### WEEK 4 Real-world time series data

- `Conv1D` layer type : add a 1D convolution to the model for predicting time series data.
- The input shape for a univariate time series to a Conv1D : [None, 1]
- `CSV` : the name of the Python library used to read CSVs.
- `next(reader)` : when you read a CSV file with a header that you don’t want to read into your dataset, to execute before iterating through the file using a ‘reader’ object.
- read a row from a reader and want to cast column 2 to another data type (float) : `float(row[2])` 
- sunspot seasonality : 11 or 22 years depending on who you ask
- neural network type do you think is best for predicting time series like our sunspots dataset : A combination of all of the above (RNN / LSTM, Convolutions, DNN)
- `MAE` : a good analytic for measuring accuracy of predictions for time series because it doesn’t heavily punish larger errors like square errors do.


</br>
</br>

