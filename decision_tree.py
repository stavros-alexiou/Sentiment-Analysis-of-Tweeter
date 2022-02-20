from pyexpat import model

import pandas as pd
df = pd.read_csv("Dataset 2.csv", encoding='latin-1', names = ['target', 'ids', 'date', 'flag', 'user', 'tweet'])
print(df.head())

import nltk
# nltk.download('stopwords')

import re
def cleanText(sample):
    sample = sample.lower() # turn every letter in lower-case.
    sample = re.sub(r'http\S+', '', sample)  # remove URLs Takes in all characters except whitespace.
    sample = re.sub(r'@\S+', '', sample)  # remove user's mention
    sample = re.sub('\'', '', sample)
    sample = re.sub('[^a-z ]', '', sample) # turn each tweet that does not comfront with the rule "[^a-zA-Z]+" in a single space.
    sample = sample.split() # takes all items from the list and joins them into one string. 
    sample = " ".join(sample) # takes all items from the list and joins them into one string.
    
    return sample

# apply the above function in dataframes tweet column and restore it.
df['tweet'] = df['tweet'].apply(cleanText)
print(df.head())

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
# use label-encoder to convert 0 -> 0 and 4->1 and save the results in a new column named 'end_target'.
df['end_target'] = le.fit_transform(df['target'].values)
print(df)

# vectorization process, turn text into sequences.
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=20000, split=' ')
tokenizer.fit_on_texts(df['tweet'].values) 

# the dependent value will be marked as 0 (target=0) and 1 ((target=4)) and is stored in y variable.S
y = df['end_target']

# converting the dataframe's 'tweet' column values from text to features.
x = tokenizer.texts_to_sequences(df['tweet'].values)
x = pad_sequences(x)

print(y.shape)
print(x.shape)

from sklearn.model_selection import train_test_split
# in this case the dataset is seperated in training and testing records.
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42, test_size=0.4)

# training a decision tree classifier.
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# check the training accuracy of the.
print(model.score(x_test, y_test))

# make prediction on validation set.
print(model.predict(x_test))

# the dataset has been tested for different percentages between training and testing records. There are the results:
# 90% training - 10% testing -> 0.59753125 score.
# 80% training - 20% testing -> 0.5936875 score.
# 60% training - 40% testing -> 0.5920515625 score.