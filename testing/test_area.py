import joblib
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ML Libraries
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import tensorflow as tf


def clean_text(sample):
    sample = sample.lower()
    # sample = sample.replace()
    sample = re.sub(r'http\S+', '', sample)  # remove URL     \S takes in all characters except whitespace.
    sample = re.sub(r'@\S+', '', sample)  # remove user's mention
    sample = re.sub("[^a-zA-Z]+", " ", sample)

    sample = sample.split()
    sample = [ps.stem(s) for s in sample if s not in stops]  # stopwords removal and ing,ed etc...
    sample = " ".join(sample)
    return sample


# Load the model and the Vectorizer from the file
LR_model_from_joblib = joblib.load('LR_model.pkl')
loaded_transformer = joblib.load('tfidf.pkl')
print(LR_model_from_joblib)

# LR_model_from_joblib.predict(X_test)
stops = set(stopwords.words('english'))
ps = PorterStemmer()

test1 = ["foivos is the best", "foivos is the worst", "He is the worst of all", "best story of all time", "you are the best!", "Machine learning is fun", "Machine learning is difficult"]
result1 = ['4','0','0','4','4','4','0']
# Calling DataFrame constructor on list
test1 = pd.Series(test1)
# print(test1)
# print(type(test1))
# cleaned_test1 = (x.head()).apply(clean_text)
# print(cleaned_test1)
cleaned_test1 = test1.apply(clean_text)
print(cleaned_test1)

# Use the loaded vectorizer
test_features=loaded_transformer.transform(cleaned_test1)
# print(test_features)

# Use the loaded model to make predictions
y_Test1_lr = LR_model_from_joblib.predict(test_features)
print("Logistic Regression predictions " + str(y_Test1_lr))
print("True polarity  " + str(result1))
