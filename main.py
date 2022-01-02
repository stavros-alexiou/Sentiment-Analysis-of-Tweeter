import pandas as pd
import re
import nltk
# nltk.download('stopwords')
from keras.models import Sequential
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tensorflow as tf
import os

from sklearn.model_selection import train_test_split


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



kappa = os.path.isfile('filteredData.csv')
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if not(kappa):
    # file does not exists exists
    namelist = ['target', 'ids', 'date', 'flag', 'user', 'tweet']
    data = pd.read_csv(r"../Dataset 2.csv", encoding='latin-1', names=namelist)

    # data = data.head(10)
    # print(data.info())
    print(data['tweet'][0])

    stops = set(stopwords.words('english'))
    ps = PorterStemmer()

    data['cleaned_tweet'] = data['tweet'].apply(clean_text)
    df = pd.DataFrame(data)
    df.to_csv('filteredData.csv', sep=',', encoding='utf-8')

data = pd.read_csv(r"filteredData.csv", encoding='latin-1')
data = data.iloc[:, 1:]
data = data.astype(str)
# print(data.head())
y = pd.get_dummies(data['target']).values
corpus = data['cleaned_tweet'].values
# print(corpus)


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# cv = CountVectorizer(max_df=0.5, max_features=50000)
# X = cv.fit_transform(corpus.astype('U'))

tokenizer = Tokenizer(num_words=2000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=' ')
tokenizer.fit_on_texts(data['cleaned_tweet'].values)

X = tokenizer.texts_to_sequences(data['cleaned_tweet'].values)
X = pad_sequences(X)

# tfidf = TfidfTransformer()
# X = tfidf.fit_transform(X)



# print(X.get_params(deep=True))
# print(X[0][0])

# print(X[0])
# print(X.shape)



from keras import models
from keras.layers import Dense, Flatten, Embedding, LSTM, Dropout, Activation

# model = models.Sequential()
# model.add( Dense(16, activation='relu', input_shape=(X.shape[1],) ) )
# model.add( Dense(16, activation='relu'))
# model.add( Dense(1, activation='sigmoid'))
# model.add(Flatten())
#
# model.summary()
#
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])




# embed_dim = 128
# lstm_out = 196
# batch_size = 32
#
# model = Sequential()
# model.add(Embedding(2000, embed_dim,input_length = X.shape[1]))
# model.add(Dropout(0.2))
# # model.add(Flatten())
# model.add(LSTM(lstm_out, dropout=0.2))
# model.add(Dense(2,activation='softmax'))
# model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
# print(model.summary())

max_features = 20000
maxlen = 80 # cut texts after this number of words (among top max_features most common words)
batch_size = 32

model = Sequential()
model.add(Embedding(max_features, 128))
model.add(Dropout(0.2) ) # <- How does the dropout work?
model.add(LSTM(128, dropout=0.2)) # try using a GRU instead, for fun
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
print(model.summary())

# tf.sparse.SparseTensor(X,X.values,X.shape)
# tf.sparse.SparseTensor(y,y.values,y.shape)

# tf.sparse.reorder(X, name=None)
# tf.sparse.reorder(y, name=None)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)



print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
print(X_train[0])
print(y_train[0])
# print(X_train[1])
# tf.sparse.reorder(X_test,name=None)
# tf.sparse.reorder(X_train, name=None)
# tf.sparse.reorder(y_test, name=None)
# tf.sparse.reorder(y_train, name=None)
# import numpy as np
# y_train = np.asarray(train_labels).astype('float32').reshape((-1,1))
# y_test = np.asarray(test_labels).astype('float32').reshape((-1,1))
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)
#
# import matplotlib.pyplot as plt
# # plt.scatter(X_train, y_train, color='black')
# plt.plot(X_train, lr.predict(X_train), color='blue', linewidth=3)
# plt.title('Tweets vs Result')
# plt.ylabel('Result')
# plt.xlabel('Tweets')

from sklearn.metrics import mean_squared_error, r2_score
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

print(lr_train_mse)
lr_results = pd.DataFrame(['Linear regression',lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']
print(lr_results)

# hist = model.fit(X_train, y_train, batch_size=128, epochs=5, validation_data=(X_test, y_test))
print()
model.fit(X_train, y_train, batch_size=batch_size, epochs=5, validation_data=(X_test, y_test))

# model.fit(X_train, y_train, batch_size=batch_size, epochs=5, verbose=False, validation_data=(X_test, y_test))

from keras.backend import clear_session
clear_session()


loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


# score, acc = model.evaluate(X_test, y_test, verbose=2, batch_size=batch_size)
# print("Score: %.2f" % (score))
# print("Validation Accuracy: %.2f" % (acc))