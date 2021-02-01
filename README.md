# GAN based data augmentation approach of multi-featured and multi-dimensional time-series wearable motion data for robust HMR with DCNN

In this paper, GAN-based data augmentation is the subject of study to use multi-feathered and multi-dimensional time-series wearable motion data for DCNN. 

For the study, vanilla GAN implemented by Pytorch open source library was used, and loss graph and output were compared to be analyzed.

Research has shown how the application of GAN works on data converted from sign to image. The results showed that there was not enough quality to be used, even if it had some effect.

However, assuming that more advanced form of DCGAN or yLSGAN is used, the output is expected to be available as the actual training data.

## 1. 서론

### 1.1 연구의 배경 및 필요성
HMR(Human Motion Recognition)은 변위센서, 속도센서, 토크센서 등 모션 인식 센서로부터 얻은 데이터로부터 대상이 어떠한 동작을 취하고 있는지 알아내는 것을 말한다. HMR을 하기위한 방법에는 여러 가지가 있지만, 그 중에서 DCNN(Deep Convolution Neural Network)을 적용해 결과를 낸다고 가정한다.

DCNN(Deep Convolution Neural Network)은 합성곱(Convolution)연산을 사용하는 인공 신경망의 한 종류이며, CNN에서 neural network layer의 수를 늘려 더 깊어진 형태의 인공 신경망을 의미한다.

DCNN의 model을 학습시키기 위해서 행렬 형태의 이미지가 train data로써 입력된다. 이 때, 훈련할 충분한 학습 데이터를 확보하지 않으면, model의 성능을 저해하는 overfitting 문제가 발생할 수 있다. 따라서 정확한 DCNN model을 만들기 위해서는 양질의 데이터를 대량으로 확보할 수 있어야 한다. 학습 데이터를 만들기 위해 기본적으로 실험 데이터를 얻어와 전처리(preprocessing) 하지만, 비용적인 문제 때문에 실험만으로 모든 데이터를 얻는 데에는 무리가 있다. 대상으로 삼을 시계열 time-series weable motion data 역시, 실험 시간이 오래 걸리는 점, 매 실험마다 데이터에 통일성을 부여하기 위해 전처리를 따로 해야한다는 점 등등 시간과 비용적인 문제와 직면한다.
#### 1.1.1 연구의 배경
#### 1.1.2 연구의 필요성
#### 1.2 연구의 목표 및 방법

## 2. 배경 이론
### 2.1 GAN(Generative Adversarial Network)
#### 2.1.1 GAN의 개념
#### 2.1.2 GAN의 학습과정
#### 2.1.3 GAN의 loss function
### 2.2 Input Parameters
#### 2.2.1 Learning Rate
#### 2.2.2 Batch Size
#### 2.2.3 Epoch
### 2.3 Training Dataset

## 3. Data augmentation
### 3.1 Case01 : Initial Condition
### 3.2 Case02 : Changed Learning rate of G & D
### 3.3 Case03 : Changed Learning rate of G
### 3.4 Case04 : Changed Batch Size
### 3.5 Final Case : Changed Epochs
#### 3.5.1 구간 분석
#### 3.5.2 Training data & Output 비교
##### 3.5.2.1 Training data
##### 3.5.2.2 Output

## 4. 결 론
