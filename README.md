# Real-or-Not-NLP-with-Disaster-Tweets

## Project Name: NLP DISASTER TWEETS: EDA, NLP, TENSORFLOW, KERAS

### Description: 

Sentiment Analysis of the dataset of twitter disaster tweets and predicting 
  1. Actual Disaster
  2. Metaphorically Disaster.
Ranking - Top 7% as of 02-05-2020

### Table of Contents:

Introduction<br>
Libraries<br>
Loading Data<br>
Exploratory Data Analysis<br>
 &nbsp; Analyzing Labels<br>
  &nbsp;Analyzing Features<br>
    &nbsp;&nbsp;Sentence Length Analysis<br>
Data Cleaning<br>
  &nbsp; Remove URL<br>
 &nbsp; Handle Tags<br>
 &nbsp; Handle Emoji<br>
&nbsp;  Remove HTML Tags<br>
&nbsp;  Remove Stopwords and Stemming<br>
&nbsp;  Remove Useless Characters<br>
&nbsp;  WORLDCLOUD<br>
Final Pre-Processing Data<br>
Machine Learning<br>
 &nbsp; Logistic Regression<br>
 &nbsp; Navie Bayes<br>
&nbsp;&nbsp;    Gaussian Naive Bayes<br>
 &nbsp;&nbsp;   Bernoulli Naive Bayes<br>
 &nbsp;&nbsp;   Complement Naive Bayes<br>
 &nbsp;&nbsp;   Multinomial Naive Bayes<br>
 &nbsp; Support Vector Machine (SVM)<br>
  &nbsp;&nbsp;  RBF kernel SVM<br>
 &nbsp;&nbsp;   Linear Kernel SVM<br>
 &nbsp; Random Forest<br>
Deep Learning<br>
&nbsp;  Single Layer Perceptron<br>
&nbsp;  Multi Layer Perceptron<br>
&nbsp;&nbsp;    Model 1 : SIGMOID + ADAM<br>
&nbsp;&nbsp;    Model 2 : SIGMOID + SGD<br>
&nbsp;&nbsp;    Model 3 : RELU + ADAM<br>
&nbsp;&nbsp;    Model 4 : RELU + SGD<br>
&nbsp;&nbsp;    Model 5 : SIGMOID + BATCH NORMALIZATION + ADAM<br>
&nbsp;&nbsp;    Model 6 : SIGMOID + BATCH NORMALIZATION + SGD<br>
&nbsp;&nbsp;    Model 7 : RELU + DROPOUT + ADAM<br>
&nbsp;&nbsp;    Model 8 : RELU + DROPOUT + SGD<br>


### Pre-requisites and Installation:
This project requires **Python** and the following Python libraries installed:
[NumPy](http://www.numpy.org/)<br>
[Pandas](http://pandas.pydata.org/)<br>
[Matplotlib](http://matplotlib.org/)<br>
[scikit-learn](http://scikit-learn.org/stable/)<br>
[Tensorflow](https://www.tensorflow.org/)<br>
[Keras](https://keras.io/)<br>
![Requirements](https://user-images.githubusercontent.com/34357926/105755591-87d8af00-5f71-11eb-9bc1-865615ff5759.png)<br>

### Usage:

#### Use the main.py file (contains all the subcodes combined.)

In a terminal or command window, navigate to the top-level project directory `Multivariate-Time-series-classification/` (that contains this README) and run command in sequence:

```bash
python anyone_file_from_objective_1.py
```

_This may take time, then do_
```bash
python files_from_objective_2.py
```

_either greedy or SI method files only one by one, then do_
```bash
python anyone_file_from_objective_3.py
```

_This is final early classififcation with earliness and accuracy._

### Results

![Datasets](https://user-images.githubusercontent.com/34357926/102748576-6e4bb280-4388-11eb-8ff0-2376ef519a85.png)

![Summary Part 2](https://user-images.githubusercontent.com/34357926/102748577-6ee44900-4388-11eb-814f-fa8986ba208f.png)

![Summary Part 3](https://user-images.githubusercontent.com/34357926/102748573-6c81ef00-4388-11eb-900c-efb769a60829.png)


### Credits:

Project is based on the paper "[Early classification on multivariate time series](https://dl.acm.org/citation.cfm?id=2841855)". Author Guoliang He, Yong Duan, Rong Peng, Xiaoyuan Jing, Tieyun Qian, Lingling Wang.



### License:

To cite either a computer program or piece of source code you will need the following information:

Yash Gupta<br />Early Classification of Time Series Data<br />https://github.com/erYash15/Multivariate-Time-Series-Early-Classification
