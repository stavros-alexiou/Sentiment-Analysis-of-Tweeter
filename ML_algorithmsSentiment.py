from preprocessing import preprocessSentiment

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ML Libraries
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier, plot_importance
from sklearn import svm, preprocessing

from joblib import dump
from matplotlib import pyplot
import joblib

from sklearn.svm import SVC
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

data = preprocessSentiment()

# LabelEncoding 0 , 2 , 4 -> 0 , 1 , 2
le = preprocessing.LabelEncoder()
y = le.fit_transform(data['sentiment'].values)

# Vectorizing the data. String -> NumVectors
corpus = data['cleaned_tweet'].values
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
dump(vectorizer, 'tfidf.pkl') # Saving the Vectorizer for later use of the saving model
# print(X)
print(corpus)
print(y[:10])

# Split dataset into Train, Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=90, shuffle=True)

# Training Naive Bayes model
NB_model = MultinomialNB()
NB_model.fit(X_train, y_train)
y_predict_nb = NB_model.predict(X_test)
print("Naive Bayes model ")
print("Accuracy score of test")
print(accuracy_score(y_test, y_predict_nb))

# Save the model as a pickle in a file
joblib.dump(NB_model, 'ML/NB_modelSent.pkl')

# Training Logistics Regression model
LogReg_model = LogisticRegression(solver='liblinear', C=100)
LogReg_model.fit(X_train, y_train)
y_predict_lr = LogReg_model.predict(X_test)
print("Logistics Regression model ")
# print(y_predict_lr)
# print(y_test)
print("Accuracy score of test")
print(accuracy_score(y_test, y_predict_lr))
y_predict_lr_train = LogReg_model.predict(X_train)
print("Accuracy score of trained")
print(accuracy_score(y_train, y_predict_lr_train))

# Save the model as a pickle in a file
joblib.dump(LogReg_model, 'ML/LR_modelSent.pkl')

# Load the model from the file
# LR_model_from_joblib = joblib.load('ML/LR_modelSent.pkl')
# print(LR_model_from_joblib)

#Create a svm Classifier
# clf = svm.SVC(kernel='linear') # Linear Kernel
# fit model no training data


XGB_model = XGBClassifier()
XGB_model.fit(X_train, y_train)
# plot feature importance
plot_importance(XGB_model)
pyplot.show()

# make predictions for test data and evaluate
predictions = XGB_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.4f%%" % (accuracy * 100.0))

# Save the model as a pickle in a file
joblib.dump(XGB_model, 'ML/XGB_modelSent.pkl')
