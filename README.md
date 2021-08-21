# Supervised Machine Learning Homework - Predicting Credit Risk

In this assignment, you will be building a machine learning model that attempts to predict whether a loan from LendingClub will become high risk or not. 

## Background

LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API.

You will be using this data to create machine learning models to classify the risk level of given loans. Specifically, you will be comparing the Logistic Regression model and Random Forest Classifier.

## Instructions

### Retrieve the data

In the `Generator` folder in `Resources`, there is a [GenerateData.ipynb](/Resources/Generator/GenerateData.ipynb) notebook that will download data from LendingClub and output two CSVs: 

* `2019loans.csv`
* `2020Q1loans.csv`

You will be using an entire year's worth of data (2019) to predict the credit risk of loans from the first quarter of the next year (2020).

Note: these two CSVs have been undersampled to give an even number of high risk and low risk loans. In the original dataset, only 2.2% of loans are categorized as high risk. To get a truly accurate model, special techniques need to be used on imbalanced data. Undersampling is one of those techniques. Oversampling and SMOTE (Synthetic Minority Over-sampling Technique) are other techniques that are also used.

## Preprocessing: Convert categorical data to numeric

Create a training set from the 2019 loans using `pd.get_dummies()` to convert the categorical data to numeric columns. Similarly, create a testing set from the 2020 loans, also using `pd.get_dummies()`. Note! There are categories in the 2019 loans that do not exist in the testing set. If you fit a model to the training set and try to score it on the testing set as is, you will get an error. You need to use code to fill in the missing categories in the testing set. 

## Consider the models

You will be creating and comparing two models on this data: a logistic regression, and a random forests classifier. Before you create, fit, and score the models, make a prediction as to which model you think will perform better. You do not need to be correct! Write down (in markdown cells in your Jupyter Notebook or in a separate document) your prediction, and provide justification for your educated guess.

## Fit a LogisticRegression model and RandomForestClassifier model

Create a LogisticRegression model, fit it to the data, and print the model's score. Do the same for a RandomForestClassifier. You may choose any starting hyperparameters you like. Which model performed better? How does that compare to your prediction? Write down your results and thoughts.

## Revisit the Preprocessing: Scale the data

The data going into these models was never scaled, an important step in preprocessing. Use `StandardScaler` to scale the training and testing sets. Before re-fitting the LogisticRegression and RandomForestClassifier models on the scaled data, make another prediction about how you think scaling will affect the accuracy of the models. Write your predictions down and provide justification.

Fit and score the LogisticRegression and RandomForestClassifier models on the scaled data. How do the model scores compare to each other, and to the previous results on unscaled data? How does this compare to your prediction? Write down your results and thoughts.

## Prediction

My prediction on which model will perform better on the unscaled data and the scaled data between Logistic Regression or Random Forest was close to a 50-50 decision. The main reason for this is, because most of the machine learning algorithms are sensitive and perform better with feature scaling. For example, Logistic Regression is used for the binary categorization of data and uses gradient descent as an optimization technique require data to be scaled. In the other hand, Random Forests are fairly insensitive to the scale of the features, because it takes the best features for our models and algorithms, by taking these insights from the data, and without the need to use expert knowledge or other kinds of external information.
So, for the previous reasons my prediction for which model will perform better for unscale data will be Random Forest, and for the scaled data my prediction is Logistic Regression model.

## Conclusion

As conclusion, looking at the scores to validate the models used for this data, I can say that the best model out of the four in this analysis to predict Credit Risk, is the Logistic Regression in scaled data with a Training Data Score: 0.7135467980295567 and Testing Data Score: 0.7003402807316036 because it looks like they are very close in value and over 70% accurate.
Secondly, I think the model of Logistic Regression in unscaled data was not as good as the previous one with a Training Data Score: 0.6959770114942528 and Testing Data Score: 0.5603998298596342 the accuracy is just 56% and the values are far apart.
Lastly, I can notice that the scores for both of the Random Forest models (scaled and unscaled) ended up being almost the same with Training Score: 1.0 and Testing Score: 0.6193109315185028, which probably means that the feature selection was equal in both models.
