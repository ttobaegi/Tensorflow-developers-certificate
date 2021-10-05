# Sequences, Time Series and Prediction

> ### Objectives 
> [텐서플로우 자격증 공식 가이드](https://www.tensorflow.org/extras/cert/TF_Certificate_Candidate_Handbook_ko.pdf?hl=ko)
> - TensorFlow 에서 시계열 및 예측 문제를 해결하는 방법을 이해해야한다.
>   - 시계열, 시퀀스, 예상 모델을 훈련, 조정 및 활용.
>   - 모델을 학습하여 일변량 및 다변량 시계열 모두에 대한 값 예측.
>   - 시계열을 학습하기 위한 데이터 준비.
>   - 평균 절대 오차(MAE)를 이해하고 시퀀스 모델의 정확성을 평가하는 데 어떻게 평균 절대 오차가 사용되는지 이해.
>   - 시계열, 시퀀스, 예상 모델의 RNN 및 CNN 활용.
>   - 중앙 창과 후행 창을 언제 사용해야 하는지 파악.
>   - 예측을 위해 TensorFlow 활용.
>   - 기능 및 라벨 준비.
>   - 시퀀스 편향 파악 및 보완.
>   - 시계열, 시퀀스, 예상 모델에서 동적으로 학습율을 조정.



> ### Table of Contents
> - [1 - Sequences and Prediction](#1)
>   - [Time Series](#1-1)
>   - [Train / Validation / Test](#1-2)
>   - [Metrics](#1-3)
>   - [Naive Forecast](#1-4)
>   - [Moving average and differencing](#1-5)
> - [2 - Deep Neural Networks for Time Series](#2)
>   - [Preparing features and labels](#2-1)
>   - [Sequence bias](#2-2)
>   - [Single layer NN](#2-3)
>   - [Deep neural network](#2-4)

</br>
</br>

<a name='1'></a>
## 1 - Sequences and Prediction
[Time-series](https://www.tensorflow.org/tutorials/structured_data/time_series)

#### Common patterns in Time-series
시계열 자료가 가지는 여러 특성들을 활용해 시계열 데이터를 생성하고 예측해보자.
- **Trend** : a specific direcion that they're moving in.
- **Seasonality** : patterns repeat at predictable intervals
- **Autocorrelation**
- **Noise**
- **Stationarity** 정상성 
  - Stationary time-sereis 
  
  ![](https://images.velog.io/images/findingflow/post/8d4b4937-0b1f-4ff7-84fa-e471b2382c03/image.png)
   - Non-stationary time-sereis 
   
  ![](https://images.velog.io/images/findingflow/post/d810f9a7-d539-4086-9373-247757cb5a9c/image.png)
- Autocorrelation 자기회귀 

  ![](https://images.velog.io/images/findingflow/post/c2a03bc8-8d43-4fb3-97f1-7c7ab15738aa/image.png)
  - Multiple autocorrelation
 
  ![](https://images.velog.io/images/findingflow/post/daecd98c-51e4-4105-926c-bd47773990c8/image.png)
- Combination of trend & seasonality & autocorrelation & noise

  ![](https://images.velog.io/images/findingflow/post/6ba06d14-5598-409f-aaca-d9b94af27b62/image.png)


앞서 살펴본 특성을 가진 시계열 데이터를 생성해보자.

```py
# modules
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt

# Time-series graph
def plot_series(time, series, format='-', start = 0, end=None):
  plt.plot(time[start:end], series[start:end], format)
  plt.xlabel("Time")
  plt.ylabel("Value")
  plt.grid(True)
```
#### Trend

```py
## trend 
def trend( time, slope=0 ) :
  return slope * time

## time
# np.arange(num) num까지의 정수가 포함된 벡터 생성
time = np.arange( 4 * 365 + 1)
print( 'shape of time :', time.shape)
print(time)
shape of time : (1461,)
[   0    1    2 ... 1458 1459 1460]

## series (value)
series = trend(time, 0.1) # time * 0.1
print(series)
[0.000e+00 1.000e-01 2.000e-01 ... 1.458e+02 1.459e+02 1.460e+02]

## time series graph
plt.figure(figsize = (10,6))
plot_series(time, series)
plt.show()
```
![](https://images.velog.io/images/findingflow/post/552f27a8-48e1-41c9-889a-916fa4ea8ef6/image.png)

#### Seasonality
```py
def seasonal_pattern(season_time) :
  # arbitrary pattern 
  return np.where(season_time < 0.1,
                  np.cos(season_time * 7 * np.pi),
                  1/np.exp( 5 * season_time ))

def seasonality( time, period, amplitude=1, phase=0 ) :
  # repeats the same pattern at each period
  season_time = ((time+phase) % period) / period
  return amplitude * seasonal_pattern(season_time)


AMPLITUDE = 40
series = seasonality(time, period = 365, amplitude = AMPLITUDE)

plt.figure(figsize = (10,6))
plot_series(time, series)
plt.show()
```
![](https://images.velog.io/images/findingflow/post/e45a8b7d-58ff-47be-b1ec-c60b8e7079d6/image.png)


#### Combinition of both trend and seasonality
```py
## trend with seasonality
slope = 0.05
baseline = 10
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude = AMPLITUDE)

plt.figure(figsize = (10,6))
plot_series(time, series)
plt.show()
```
![](https://images.velog.io/images/findingflow/post/cc5e9fd6-2054-4e61-b497-69946f70eae6/image.png)

#### Noise
```py
## noise
def white_noise(time, noise_level=1, seed=None):
  rnd = np.random.RandomState(seed)
  return rnd.randn(len(time)) * noise_level
  
noise_level = 5
noise = white_noise(time, noise_level, seed=42)

plt.figure(figsize = (10,6))
plot_series(time, noise)
plt.show()
```
![](https://images.velog.io/images/findingflow/post/1f9db980-376b-4511-8ebf-164761b36798/image.png)
```py
# series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude = AMPLITUDE)
series += noise

plt.figure(figsize = (10,6))
plot_series(time, series)
plt.show()
```
![](https://images.velog.io/images/findingflow/post/ce2ff1ce-2284-4904-8e44-2af1fe1397fe/image.png)

</br>
