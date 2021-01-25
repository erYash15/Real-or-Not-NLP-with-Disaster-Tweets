# Real-or-Not-NLP-with-Disaster-Tweets





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
Data Cleaning
  &nbsp; Remove URL
 &nbsp; Handle Tags
 &nbsp; Handle Emoji
&nbsp;  Remove HTML Tags
&nbsp;  Remove Stopwords and Stemming
&nbsp;  Remove Useless Characters
&nbsp;  WORLDCLOUD

Final Pre-Processing Data

Machine Learning
 &nbsp; Logistic Regression
 &nbsp; Navie Bayes
&nbsp;&nbsp;    Gaussian Naive Bayes
 &nbsp;&nbsp;   Bernoulli Naive Bayes
 &nbsp;&nbsp;   Complement Naive Bayes
 &nbsp;&nbsp;   Multinomial Naive Bayes
 &nbsp; Support Vector Machine (SVM)
  &nbsp;&nbsp;  RBF kernel SVM
 &nbsp;&nbsp;   Linear Kernel SVM
 &nbsp; Random Forest

Deep Learning
&nbsp;  Single Layer Perceptron
&nbsp;  Multi Layer Perceptron
&nbsp;&nbsp;    Model 1 : SIGMOID + ADAM
&nbsp;&nbsp;    Model 2 : SIGMOID + SGD
&nbsp;&nbsp;    Model 3 : RELU + ADAM
&nbsp;&nbsp;    Model 4 : RELU + SGD
&nbsp;&nbsp;    Model 5 : SIGMOID + BATCH NORMALIZATION + ADAM
&nbsp;&nbsp;    Model 6 : SIGMOID + BATCH NORMALIZATION + SGD
&nbsp;&nbsp;    Model 7 : RELU + DROPOUT + ADAM
&nbsp;&nbsp;    Model 8 : RELU + DROPOUT + SGD


### Pre-requisites and Installation:
This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [Matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [Tensorflow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)

![Requirements](https://user-images.githubusercontent.com/34357926/105755591-87d8af00-5f71-11eb-9bc1-865615ff5759.png)

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
