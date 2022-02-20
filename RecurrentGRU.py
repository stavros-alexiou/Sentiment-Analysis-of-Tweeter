from preprocessing import preprocess
from preprocessing import preprocessSentiment

import pandas as pd
import re
import nltk
# nltk.download('stopwords')
from keras.models import Sequential
import tensorflow as tf
import os

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, Dropout, CuDNNGRU
from numpy import loadtxt
import pickle

# choose between Given or Sentiment Dataset
# Target 0 4
data = preprocess()
y = pd.get_dummies(data['target']).values

# Target 0 2 4
# data = preprocessSentiment()
# y = pd.get_dummies(data['sentiment']).values

corpus = data['cleaned_tweet'].values

# a simple tokenization of the data in order to vectorize.Tokenizer instead of Vectorizer we want to use Embedding layer
tokenizer = Tokenizer(num_words=5000,  split=' ')
tokenizer.fit_on_texts(data['cleaned_tweet'].values)

# Transforms each text in texts to a sequence of integers. vectorization
X = tokenizer.texts_to_sequences(data['cleaned_tweet'].values)
X = pad_sequences(X)

# saving tokenizer for later use
with open('testing/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True)



max_features = 5000
batch_size = 256


### First GRU Neural Network ###
## you can uncomment and run

# model = Sequential()
# model.add(Embedding(input_dim=max_features, output_dim=batch_size, input_length=X.shape[1]))
# model.add(Dropout(0.2))
# model.add(CuDNNGRU(batch_size) ) # , activation='sigmoid')) # try using a GRU instead, for fun
# model.add(Dense(2, activation='softmax'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
#
# hist = model.fit(X_train, y_train, batch_size=128, epochs=5, validation_data=(X_test, y_test))
#
# # save model and architecture to single file
# model.save("testing/NeuralNetworks/GRU.h5")
# print("Saved GRU model to disk")
#
# loss, accuracy = model.evaluate(X_train, y_train)
# print("Training Accuracy: {:.4f}".format(accuracy))
# print("Training Loss: {:.4f}".format(loss))
# lossT, accuracyT = model.evaluate(X_test, y_test)
# print("Testing Accuracy:  {:.4f}".format(accuracyT))
# print("Testing Loss: {:.4f}".format(lossT))


### Second GRU Neural Network ###
## This Neural Network was used for testing

model = Sequential()
# Embedding Layer turns positive integers (indexes) into dense vectors of fixed size. Widely used on text data!
model.add(Embedding(input_dim=max_features, output_dim=batch_size, input_length=X.shape[1]))
model.add(Dropout(0.2))        # Randomly drop out 20% of layer outputs in order to prevent Overffiting
model.add(CuDNNGRU(batch_size, return_sequences=True))  # Return the full sequence in the output sequence.
model.add(Dropout(0.2))
model.add(CuDNNGRU(batch_size))
# model.add(Dense(2, activation='softmax')) # dataset target 0, 4. preprocess() function
model.add(Dense(3, activation='softmax'))   # dataset target 0, 2, 4. preprocessSentiment() function

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

hist = model.fit(X_train, y_train, batch_size=128, epochs=5, validation_data=(X_test, y_test))


# save model and architecture to single file
model.save("testing/NeuralNetworks/RNNGRU.h5")
print("Saved RecurciveGRU model to disk")

# # load and evaluate a saved model
# from numpy import loadtxt
# from keras.models import load_model
#
# # load model
# model = load_model('model.h5')
# # summarize model.
# model.summary()
# # load dataset
# dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
# # split into input (X) and output (Y) variables
# X = dataset[:,0:8]
# Y = dataset[:,8]
# # evaluate the model
# score = model.evaluate(X, Y, verbose=0)


loss, accuracy = model.evaluate(X_train, y_train)
print("Training Accuracy: {:.4f}".format(accuracy))
print("Training Loss: {:.4f}".format(loss))
lossT, accuracyT = model.evaluate(X_test, y_test)
print("Testing Accuracy:  {:.4f}".format(accuracyT))
print("Testing Loss: {:.4f}".format(lossT))




