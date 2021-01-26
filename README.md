# Real-or-Not-NLP-with-Disaster-Tweets

## Project Name: NLP DISASTER TWEETS: EDA, NLP, TENSORFLOW, KERAS

## Description: 
Sentiment Analysis of the dataset of twitter disaster tweets and predicting<br>
&nbsp; Actual Disaster<br>
&nbsp; Metaphorically Disaster<br>

## Table of Contents:

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


## Pre-requisites and Installation:
This project requires **Python** and the following Python libraries installed:<br>
&nbsp;&nbsp; [NumPy](http://www.numpy.org/)<br>
&nbsp;&nbsp; [Pandas](http://pandas.pydata.org/)<br>
&nbsp;&nbsp; [Matplotlib](http://matplotlib.org/)<br>
&nbsp;&nbsp; [scikit-learn](http://scikit-learn.org/stable/)<br>
&nbsp;&nbsp; [Tensorflow](https://www.tensorflow.org/)<br>
&nbsp;&nbsp; [Keras](https://keras.io/)<br><br>
![Requirements](https://user-images.githubusercontent.com/34357926/105755591-87d8af00-5f71-11eb-9bc1-865615ff5759.png)<br>

## Data Overview

Size of tweets.csv - 1.53MB<br>
Number of rows in tweets.csv = 11369<br>
**Features:**<br>
id - a unique identifier for each tweet<br>
text - the text of the tweet<br>
location - the location the tweet was sent from (may be blank)<br>
keyword - a particular keyword from the tweet (may be blank)<br>
target - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)<br>

## WordCloud

Word Clouds are a visual representation of the frequency of words within a given tweets.
![Word Cloud](https://user-images.githubusercontent.com/34357926/105754188-c7060080-5f6f-11eb-9122-71fc6319c040.PNG)

## Results

### Key Performance Index:

**Micro f1 score**: Calculate metrics globally by counting the total true positives, false negatives and false positives. This is a better metric when we have class imbalance.<br>
**Macro f1 score**: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.<br>
**Micro-Averaged F1-Score (Mean F Score)**: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:<br>
F1 = 2 (precision recall) / (precision + recall)

All the models are compared on the basis of Accuracy, Precision, Recall, F1-Score, Time. <br>

![Results](https://user-images.githubusercontent.com/34357926/105753395-a2f5ef80-5f6e-11eb-8d3e-cfda9f9c630b.png)

Best Performing Models are: - Support Vector Machine, Deep Learning(Relu + Adam), Deep Learning(Relu + Adam + Dropouts)<br>


## Conclusion

Deep Learning Models are easy to overfit and underfit.<br>
Do not underestimate the power of Machine Learning techniques.<br>
Relu and Adam with Dropout proved to best as expected.<br>
SVM is still the best as far as accuracy and training time is concerned.


## References:

- https://www.kaggle.com/vbmokin/nlp-with-disaster-tweets-cleaning-data
- https://towardsdatascience.com/natural-language-processing-nlp-for-machine-learning-d44498845d5b
- https://machinelearningmastery.com/

