#The dataset has only values 0 and 4, but the values range is 0,2,4 district

import pandas as pd
import re
import matplotlib.pyplot as plt
from keras import models
from keras.layers import Embedding, LSTM, Dense, Dropout
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix


header_list = ["Target", "Id", "Date", "Flag", "User", "Tweet"]

sw = set(stopwords.words('english'))


def clean_text(sample):
    sample = sample.lower()
    sample = re.sub("@[A-Za-z0-9_]+", "", sample)#remove mentions
    sample = re.sub("#[A-Za-z0-9_]+", "", sample)#remove hashtages because i want to the get the clean content of a tweet without all these elements
    sample = re.sub(r"http\S+", "", sample)
    sample = re.sub(r"www.\S+", "", sample)
    sample = re.sub('[()!?]', ' ', sample)
    sample = re.sub('\[.*?\]', ' ', sample)
    sample = sample.split()
    sample = [s for s in sample if s not in sw]
    sample = ' '.join(sample)
    return sample

def model_construction(data_train):
    model = models.Sequential()
    model.add(Dense(128, input_shape=[data_train.shape[1]], activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


df = pd.read_csv("Dataset 2.csv", encoding='latin-1', on_bad_lines='warn', names=header_list)

df['cleared_summaries'] = df['Tweet'].apply(clean_text)

df['Target'] = [1 if x == 4 else 0 for x in df.Target]


corpus = df['cleared_summaries'].values

Y = df['Target'].values

cv = CountVectorizer(max_df = 0.6)
tfidf = TfidfTransformer()


X = cv.fit_transform(corpus)
X = tfidf.fit_transform(X)

X = csr_matrix.sorted_indices(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state = 1)

model1 = model_construction(X_train)

model1.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=1024, epochs=10)

predict = model1.predict(X_test)

predict[predict >= 0.5] = 1

predict = predict.astype('int')

dic = {0: 'neg', 1:'pos'}

predict = [dic[p[0]] for p in predict]

output = pd.DataFrame(data={"review": X_test, "Test_value":y_test, "Prediction":predict})
print(output)
# output.to_csv("model_result.csv")



