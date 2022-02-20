from preprocessing import preprocess

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing

# ML Libraries
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance
from sklearn import svm

from joblib import dump
from matplotlib import pyplot
import joblib

from sklearn.svm import SVC
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

data = preprocess()

# LabelEncoding 0 , 4 -> 0 , 1
le = preprocessing.LabelEncoder()
y = le.fit_transform(data['target'].values)
print(y)

# Vectorizing the data. String -> NumVectors
corpus = data['cleaned_tweet'].values
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
dump(vectorizer, 'testing/tfidf.pkl')
# print(X)
# print(corpus)
# print(y)

# Split dataset into Train, Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=90, shuffle=True)

# Training Naive Bayes model
NB_model = MultinomialNB()
NB_model.fit(X_train, y_train)
y_predict_nb = NB_model.predict(X_test)
print("Naive Bayes model ")
print(accuracy_score(y_test, y_predict_nb))

# Save the model as a pickle in a file
joblib.dump(NB_model, 'testing/NB_model.pkl')

# Training Logistics Regression model
LogReg_model = LogisticRegression(solver='liblinear', C=100)
LogReg_model.fit(X_train, y_train)
y_predict_lr = LogReg_model.predict(X_test)
print("Logistics Regression model ")
print("Accuracy score of test")
print(accuracy_score(y_test, y_predict_lr))
y_predict_lr_train = LogReg_model.predict(X_train)
print("Accuracy score of trained")
print(accuracy_score(y_train, y_predict_lr_train))
# Logistics Regression model
# Accuracy score of test
# 0.7835    0.7833625   0.780075

# Save the model as a pickle in a file
joblib.dump(LogReg_model, 'testing/LogReg_model.pkl')

# Load the model from the file
LR_model_from_joblib = joblib.load('testing/LR_model.pkl')
print(LR_model_from_joblib)

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel
# clf.fit(X_train, y_train)
# # make predictions for test data and evaluate
# predictions = clf.predict(X_test)
# accuracy = accuracy_score(y_test, predictions)
# print("SVM model ")
# print("Accuracy: %.4f%%" % (accuracy * 100.0))
# # Save the model as a pickle in a file
# joblib.dump(clf, 'SVM_model.pkl')

# fit model no training data
XGB_model = XGBClassifier()
XGB_model.fit(X_train, y_train)
# plot feature importance
# plot_importance(XGB_model)
# pyplot.show()

# make predictions for test data and evaluate
predictions = XGB_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("XGB model ")
print("Accuracy: %.4f%%" % (accuracy * 100.0))
# Accuracy: 75.98%  75.98% 76.095


# Save the model as a pickle in a file
joblib.dump(XGB_model, 'testing/XGB_model.pkl')
