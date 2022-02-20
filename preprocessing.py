import pandas as pd
import re
import os
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords



def clean_text(sample):
    sample = sample.lower()                 # lowercase
    # sample = sample.replace()
    sample = re.sub(r'http\S+', '', sample) # remove URL     \S takes in all characters except whitespace.
    sample = re.sub(r'@\S+', '', sample)    # remove user's mention
    sample = re.sub('\'', '', sample)       # remove 's, 'm, 't == cant't-> cant
    sample = re.sub('[^a-z ]', '', sample)

    sample = sample.split()
    # sample = [ps.stem(s) for s in sample if s not in stops]  # stopwords removal and ing,ed etc...
    sample = " ".join(sample)
    return sample


def preprocess():
    kappa = os.path.isfile('filteredData.csv')
    if not kappa:
        # cleaned dataset does not exists
        namelist = ['target', 'ids', 'date', 'flag', 'user', 'tweet']
        data = pd.read_csv(r"Dataset 2.csv", encoding='latin-1', names=namelist)

        # stemming wasn't helping accuracy; on the contrary
        # stops = set(stopwords.words('english'))
        # ps = PorterStemmer()

        data['cleaned_tweet'] = data['tweet'].apply(clean_text)
        # keep only the columns 'target' and 'cleaned_tweet'
        data = data.drop(['ids', 'date', 'flag', 'user', 'tweet'], axis=1)
        df = pd.DataFrame(data)
        df.to_csv('filteredData.csv', sep=',', encoding='utf-8')
        return df
    else:
        # cleaned dataset exists
        df = pd.read_csv(r"filteredData.csv", encoding='utf-8')
        df = df.iloc[:, 1:]
        df = df.astype(str)
        return df


def preprocessSentiment():
    sia = SentimentIntensityAnalyzer()

    score = sia.polarity_scores("What a bad day")
    # example of Sentiment Analyzer
    print(score)

    giota = os.path.isfile('filteredSentimentData.csv')
    if not giota :
        # cleaned dataset does not exists
        namelist = ['target', 'ids', 'date', 'flag', 'user', 'tweet']
        data = pd.read_csv(r"../Dataset 2.csv", encoding='latin-1', names=namelist)

        # lexicon-based sentiment analysis of the dataset
        data['scores'] = data['tweet'].apply(sia.polarity_scores)
        # print(data['scores'])

        data['compound'] = data['scores'].apply(lambda scores: scores['compound'])
        data['sentiment'] = ''
        data.loc[data.compound > 0, 'sentiment'] = 4
        data.loc[data.compound == 0, 'sentiment'] = 2
        data.loc[data.compound < 0, 'sentiment'] = 0

        # stemming wasn't helping accuracy; on the contrary
        # stops = set(stopwords.words('english'))
        # ps = PorterStemmer()

        data['cleaned_tweet'] = data['tweet'].apply(clean_text)
        # keep only the columns 'sentiment' and 'cleaned_tweet'
        data = data.drop(['target', 'ids', 'date', 'flag', 'user', 'tweet', 'scores', 'compound'], axis=1)
        print(data)
        df = pd.DataFrame(data)
        df.to_csv('filteredSentimentData.csv', sep=',', encoding='utf-8')
        return df
    else:
        # cleaned dataset exists
        df = pd.read_csv(r"filteredSentimentData.csv", encoding='utf-8')
        df = df.iloc[:, 1:]
        df = df.astype(str)
        return df


# data = preprocess()
# print(data)
# dataSent = preprocessSentiment()
# print(dataSent)

