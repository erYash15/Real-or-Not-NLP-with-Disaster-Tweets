# Real-or-Not-NLP-with-Disaster-Tweets

## Project Name: NLP DISASTER TWEETS: EDA, NLP, TENSORFLOW, KERAS

### Description: 
Sentiment Analysis of the dataset of twitter disaster tweets and predicting<br>
&nbsp; Actual Disaster<br>
&nbsp; Metaphorically Disaster<br>

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
This project requires **Python** and the following Python libraries installed:<br>
&nbsp;&nbsp; [NumPy](http://www.numpy.org/)<br>
&nbsp;&nbsp; [Pandas](http://pandas.pydata.org/)<br>
&nbsp;&nbsp; [Matplotlib](http://matplotlib.org/)<br>
&nbsp;&nbsp; [scikit-learn](http://scikit-learn.org/stable/)<br>
&nbsp;&nbsp; [Tensorflow](https://www.tensorflow.org/)<br>
&nbsp;&nbsp; [Keras](https://keras.io/)<br><br>
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

### WordCloud

![Word Cloud](https://user-images.githubusercontent.com/34357926/105754188-c7060080-5f6f-11eb-9122-71fc6319c040.PNG)

### Results

All the models are compared on the basis of Accuracy, Precision, Recall, F1-Score, Time. <br>

![Results](https://user-images.githubusercontent.com/34357926/105753395-a2f5ef80-5f6e-11eb-8d3e-cfda9f9c630b.png)

Best Performing Models are: - Support Vector Machine, Deep Learning(Relu + Adam), Deep Learning(Relu + Adam + Dropouts)<br>


### Conclusion

Deep Learning Models are easy to overfit and underfit.<br>
Do not underestimate the power of Machine Learning techniques.<br>
Relu and Adam with Dropout proved to best as expected.<br>
SVM is still the best as far as accuracy and training time is concerned.


### References:


