import pandas as pd

df = pd.read_csv('filteredData.csv')

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
# use label-encoder to convert 0 -> 0 and 4->1 and save the results in a new column named 'end_target'.
df['end_target'] = le.fit_transform(df['target'].values)
print(df)

# the dependent value will be marked as 0 (target=0) and 1 ((target=4)) and is stored in y variable.
y = pd.get_dummies(df['end_target']).values
df = df.iloc[:, 1:]
df = df.astype(str)

# vectorization process, turn text into sequences.
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=50000, split=' ')
tokenizer.fit_on_texts(df['cleaned_tweet'].values)
# variable x is a sparse matrix and can not be printed directly.
x = tokenizer.texts_to_sequences(df['cleaned_tweet'].values)
x = pad_sequences(x)

# in this case the dataset is seperated in training and testing records.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

print(x_train.shape)
print(x_train[0].shape)

from keras.models import Sequential
model = Sequential()

# building the neural network.
from keras.layers import Dense, Dropout, CuDNNLSTM, Embedding
max_features = 10000

model.add(Embedding(max_features, 128)) #Turns positive integers (indexes) into dense vectors of fixed size
model.add(CuDNNLSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True)) 
model.add(Dropout(0.2)) # randomly sets input units to 0 with a frequency of 20% at each step during training time.
model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu')) # computes the dot product between the inputs and the kernel.
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
print(model.summary())

# training the neural network.
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

# print the training_accuracy.
loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))

# print the training_accuracy.
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

# make a prediction on the test data.
print(model.predict(x_test))

# save model and architecture to single file
model.save("alexiou.h5")
print("Saved model to disk")

# load and evaluate a saved model
# from numpy import loadtxt
# from keras.models import load_model

# load model
# model = load_model('model.h5')

# summarize model.
# model.summary()

# load dataset
# dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
# X = dataset[:,0:8]
# Y = dataset[:,8]

# evaluate the model
# score = model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# the dataset has been tested for different percentages between training and testing records. There are the results:
# 90% training - 10% testing -> loss = 0.2954, val_loss=0.3017, training_accuracy = 0.8977, testing_accuracy = 0.8777.
# 80% training - 20% testing -> loss = 0.2964, val_loss=0.3154, training_accuracy = 0.8855, testing_accuracy = 0.8662.
# 60% training - 40% testing -> loss = 0.2971, val_loss=0.3161, training_accuracy = 0.8879, testing_accuracy = 0.8641.