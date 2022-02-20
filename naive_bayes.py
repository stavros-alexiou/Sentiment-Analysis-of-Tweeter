from matplotlib.style import use
from sklearn.metrics import roc_auc_score

import pandas as pd
# read the dataset and import it in a Dataframe called df.
df = pd.read_csv("Dataset 2.csv", encoding='latin-1', names = ['target', 'ids', 'date', 'flag', 'user', 'tweet'])
print(df.head())

import nltk
# nltk.download('stopwords')
import re
def cleanText(sample):
    sample = re.sub(r'http\S+', '', sample)  # remove URLs. Takes in all characters except whitespace.
    sample = re.sub(r'@\S+', '', sample)  # remove user's mentions.
    sample = re.sub('\'', '', sample)
    sample = re.sub("[^a-zA-Z]+", " ", sample) # turn each tweet that does not comfront with the rule "[^a-zA-Z]+" in a single space.
    sample = sample.split() # divides the strings into a list with a white-space seperator.
    sample = " ".join(sample) # takes all items from the list and joins them into one string. 
    
    return sample

# apply the above function in dataframes tweet column and restore it.
df['tweet'] = df['tweet'].apply(cleanText)
print(df.head())

# tfidf vectorizer.
from nltk.corpus import stopwords
stopset = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer
# use inverted document's frequency, turn all letters to lower case, remove accents, remove words that match the imported stopwords list.
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
# use label-encoder to convert 0 -> 0 and 4->1 and save the results in a new column named 'end_target'.
df['end_target'] = le.fit_transform(df['target'].values)
print(df)

# the dependent value will be marked as 0 (target=0) and 1 ((target=4)) and is stored in y variable.
y = df['end_target']

# converting the dataframe's 'tweet' column values from text to features that are sotred in x variable.
x = vectorizer.fit_transform(df['tweet'])

print(y.shape)
print(x.shape)

from sklearn.model_selection import train_test_split
# in this case the dataset is seperated in training and testing records. 
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42, test_size=0.4)

# training a naive bayes classifier.
from sklearn import naive_bayes
clf = naive_bayes.MultinomialNB()
clf.fit(x_train, y_train)

# display the final score of the naive bayes algorithm in terminal.
print(roc_auc_score(y_test, clf.predict_proba(x_test)[:,1]))

# make a prediction.
import numpy as np
tweets_array = np.array(['Dataset 2.csv'])
tweets_vector = vectorizer.transform(tweets_array)
print(clf.predict(tweets_vector))

# the dataset has been tested for different percentages between training and testing records. There are the results:
# 90% training - 10% testing -> 0.8469209267301927 score.
# 80% training - 20% testing -> 0.8452052634169854 score.
# 60% training - 40% testing -> 0.8443333971307521 score.