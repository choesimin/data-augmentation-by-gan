# GAN based data augmentation approach of multi-featured and multi-dimensional time-series wearable motion data for robust HMR with DCNN

In this paper, GAN-based data augmentation is the subject of study to use multi-feathered and multi-dimensional time-series wearable motion data for DCNN. 

For the study, vanilla GAN implemented by Pytorch open source library was used, and loss graph and output were compared to be analyzed.

Research has shown how the application of GAN works on data converted from sign to image. The results showed that there was not enough quality to be used, even if it had some effect.

However, assuming that more advanced form of DCGAN or yLSGAN is used, the output is expected to be available as the actual training data.

## 1. 서론

### 1.1 연구의 배경 및 필요성

#### 1.1.1 연구의 배경
HMR(Human Motion Recognition)은 변위센서, 속도센서, 토크센서 등 모션 인식 센서로부터 얻은 데이터로부터 대상이 어떠한 동작을 취하고 있는지 알아내는 것을 말한다. HMR을 하기위한 방법에는 여러 가지가 있지만, 그 중에서 DCNN(Deep Convolution Neural Network)을 적용해 결과를 낸다고 가정한다.

DCNN(Deep Convolution Neural Network)은 합성곱(Convolution)연산을 사용하는 인공 신경망의 한 종류이며, CNN에서 neural network layer의 수를 늘려 더 깊어진 형태의 인공 신경망을 의미한다.

DCNN의 model을 학습시키기 위해서 행렬 형태의 이미지가 train data로써 입력된다. 이 때, 훈련할 충분한 학습 데이터를 확보하지 않으면, model의 성능을 저해하는 overfitting 문제가 발생할 수 있다. 따라서 정확한 DCNN model을 만들기 위해서는 양질의 데이터를 대량으로 확보할 수 있어야 한다. 학습 데이터를 만들기 위해 기본적으로 실험 데이터를 얻어와 전처리(preprocessing) 하지만, 비용적인 문제 때문에 실험만으로 모든 데이터를 얻는 데에는 무리가 있다. 대상으로 삼을 시계열 time-series weable motion data 역시, 실험 시간이 오래 걸리는 점, 매 실험마다 데이터에 통일성을 부여하기 위해 전처리를 따로 해야한다는 점 등등 시간과 비용적인 문제와 직면한다.

#### 1.1.2 연구의 필요성
데이터 증식(data augmentation)은 적은 양의 데이터에 인위적인 변화를 주어 새로운 학습 데이터를 얻어내는 것에 목적이 있으며, 여러 가지 방법이 존재한다. 그 중에서 GAN(Generative Adversarial Network) 기반의 데이터 증식 처리법은 많은 분야에서 성능을 인정받고 있으며, 어떤 분포의 data도 모방할 수 있다는 장점을 가진다.

DCNN에 활용하기 위해 time-series weable motion data를 Tensor로, Tensor를 음영 정보를 담은 pixel의 집합으로 처리하였다. 이렇게 만들어진 이미지는 시계열의 정보를 단지 이미지화하였을 뿐이므로, 육안으로 그 규칙성을 찾아내는 것은 불가능하다. 또한 규칙성을 찾을 수 없기 때문에 유효성있는 Data augmentation은 더욱 어려울 것으로 보인다.

위와 같은 상황에서 GAN은 육안으로 찾아낼 수 없는 pattern을 Deep Learning을 통해서 모델화할 수 있는 방법이 될 수 있다. performance 좋은 model을 만들어낸다면, 생성된 model을 통해 유효한 data를 얼마든지 얻어낼 수 있을 것이다.

#### 1.2 연구의 목표 및 방법
본 연구의 목표는 time-series weable motion data를 대상으로 한 Generator와 Discriminator의 성능을 보장할 수 있는 model을 생성해내는 것이다. 학습이 잘 되었는지를 판별하기 위해서는 loss graph와 생성된 data(image)를 사용한다. model의 performance를 높이기 위해서 network에 대한 적절한 parameter값을 찾는 것이 세부 목표이다.

연구는 GAN 기반의 data augmentation을 할 때, 여러 input parameter에 변화를 주는 방식으로 진행하며, 각 경우의 특징을 고찰하고 분석한다.

GAN의 종류에는 기존의 Vanilla GAN에서 각 환경에 맞추어 발전한 형태인 DCGAN, DeliGAN, MGAN, SRGAN 등등 이 밖에도 여러 가지가 있다. 그러나 본 논문에서는 input parameter 조정에 따른 model performance 변화에 중점을 두기 때문에 가장 기본적인 Vanilla GAN에 22*14 크기의 대상 data를 사용해 진행한다.

- 개발 언어 : python3.8
- Library : PyTorch
- OS : Windows10

## 2. 배경 이론

### 2.1 GAN(Generative Adversarial Network)

#### 2.1.1 GAN의 개념
생성적 적대 신경망(GAN)은 Generator와 Discriminator, 2개의 네트워크로 구성되어있다. GAN은 이 두 모델을 적대적(Adversarial)으로 경쟁시키며 발전시킨다.
Generator는 모방의 대상이 되는 Real Data를 재료 삼아 Fake Data를 생성해낸다(Generative한 역할). Discriminator는 Generator가 만들어낸 Fake Data가 가짜인지 진짜인지 구분한다.

두 모델은 이 과정을 반복하며 Generator는 Discriminator를 속이기 위한, Discriminator는 Generator에게 속지 않기 위한 방향으로 모델을 발전시키고, 따라서 model의 performance는 높아진다. 
이러한 두 네트워크의 관계는 GAN의 적대적 학습 전략의 핵심 개념이다.

#### 2.1.2 GAN의 학습과정

1. Random Noise를 생성한다. 
2. Generator가 만들어진 Noise를 이용해 Fake Data를 만들어 낸다.
3. Discriminator는 Real Data 또는 Fake Data를 입력받아 Discriminator가 판단한 ‘입력이 Real Data일 확률’을 0과 1 사이의 값으로 출력한다.
4. Generator는 Discriminator가 출력한 값을 반영하여 model을 학습하고 다시 Data를 생성한다.
5. Discriminator도 판단값을 정답에 비교하며 model에 반영한다.
6. 1~5 과정을 반복하며  학습한다.


### 2.2 Input Parameters

#### 2.2.1 Learning Rate
learning Rate(학습률)는 학습의 속도를 결정한다. 

Gradient descent(경사 하강법)은 함수의 기울기를 구해 그 절댓값이 낮은 방향으로 이동시켜 극값에 이르게 하는 근삿값 발견 최적화 알고리즘이다. learning Rate는 함수의 경사를 따라 이동하는 속도를 결정짓는 상수이며, learning Rate의 설정에 따라서 학습의 양상이 크게 달라진다. learning Rate의 값을 적절히 설정한 경우에는 극값에 수렴할 수 있지만, 적정치보다 크거나 작게 준다면 학습이 제대로 이루어지지 않는다.

learning Rate가 적정치보다 큰 경우, 최솟값을 계산하도록 수렴하지 못하고 함수값이 커지는 방향으로 최적화가 진행될 수 있다.
learning Rate가 적정치보다 작은 경우, 발산하지는 않지만 최적값을 구하는 데에 오랜 시간이 소요된다.

따라서 loss를 수렴하도록 하기 위해서는 적절한 learning Rate를 설정하는 것이 매우 중요하다.

#### 2.2.2 Batch Size
batch size는 data를 network에 넘길 때, 한 번에 넘겨주는 양을 말한다. 다른 말로는 mini batch라고도 한다.

batch size가 큰 경우, 더 안정적인 training이 가능하지만, 가용 메모리가 충분히 확보된 상태여야 한다.

batch size와 model performance의 상관관계는 아직 명확히 규명되지 않았지만, 값에 따라 performance에 차이가 나기 때문에 반복 수행으로 값을 조정해가며 대상 데이터에 대한 batch size의 최적값을 찾는 것이 일반적이다.

#### 2.2.3 Epoch
epoch은 dataset 전부에 대한 학습을 nerwork가 수행한 횟수를 의미하며, 매 epoch마다 weight값이 update된다.

epoch을 적정 횟수보다 작게 설정하면 모델이 너무 단순해져 loss가 줄어들지 않는 현상인 underfitting이 일어나고, 적정 횟수보다 크게 설정하면 optimun(최적화)를 지나쳐 overfitting(과적합)하게 된다.

epoch도 batch size와 마찬가지로 적절한 결과를 얻기 위해서는 반복된 실험이 필요하다.

### 2.3 Training Dataset
본 연구에서 사용하는 Train DataSet은 time-series weable motion data를 담고있는 image들이다. DCNN에 활용하기 위해 pre-processing하여 2214 pixel image로 변환한 것이며, 총 4가지 Label, (2, 3, 4, 5)로 분류된다.

[Fig. 2-2]는 pre-processing이 된 image를 보기 편하게 표현해놓은 것이며, 실제 data는 22*14 크기이기 때문에 육안으로는 작고 흐릿하게 보인다.


사진


[Fig. 2-3]는 time-series data를 이용하여 규칙성이 있게 그라데이션 형태를 띄도록 만들어낸 image이다. network가 각 label의 image 특징을 추출하여 규칙성을 찾아내는 것이 model performance를 좌우할 것이다.



## 3. Data augmentation
batch size, epoch, learning rate를 조정하며, training dataset에 대한 최적값을 찾는다.

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
