import sys
import warnings
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn import linear_model
from prettytable import PrettyTable
import numpy as np

warnings.filterwarnings("ignore")

STOPSET_WORDS = ['might', 'may', 'would', 'must', 'lgtm', 'could', 'can', 'good', 'great', 'nice', 'well', \
                 'better', 'worse', 'worst', 'should', 'i', "i'll", "ill", "it's", "its", "im", "i'm", \
                 "they're", "theyre", "you're", "youre", "that's", 'btw', "thats", "theres", "shouldnt", \
                 "shouldn't", "didn't", "didnt", "dont", "don't", "doesn't", "doesnt", "wasnt", "wasn't", \
                 'sense', "mon", 'monday', 'tue', 'wed', 'wednesday', 'thursday', 'lgtm', 'pinging', 'thu', 'friday', 'fri', \
                 'sat', 'saturday', 'sun', 'sunday', 'jan', 'january', 'feb', 'february', 'mar', 'march', \
                 'apr', 'april', 'may', 'jun', 'june', 'july', 'jul', 'aug', 'august', 'sep', 'september', \
                 'oct', 'october', 'nov', 'novenber', 'dec', 'december', 'pm', 'am', '//'
]


def structure(data_file_path):
    data = pd.read_csv(data_file_path, sep=",", header=None, names=['text', 'label'])
    return data


def remove_stopwords(data):
    stopset = set(stopwords.words('english'))
    for word in STOPSET_WORDS:
        stopset.add(word)

    data['text'] = data['text'].apply(lambda sentence: ' '.join([word for word in sentence.lower().split() \
                                                                 if word not in (stopset)]))
    return data


def tf_idf_vectorizer(data, train_data, test_data):
    tf_idf_vector = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    tf_idf_vector.fit(data['text'])
    train_data_tf_idf = tf_idf_vector.transform(train_data['text'])
    test_data_tf_idf = tf_idf_vector.transform(test_data['text'])

    return train_data_tf_idf, test_data_tf_idf


def oversample(X, Y):
    sm = SMOTE(random_state=42)

    return sm.fit_resample(X, Y)


def train(classifier, train_data, train_label):
    trained_classifier = classifier.fit(train_data, train_label)

    return trained_classifier


train_data_location = "files/dataset/Brunet2014.csv"
test_data_location = sys.argv[1]+"/comments.csv"

train_data = structure(train_data_location)
test_data = structure(test_data_location)

train_data = remove_stopwords(train_data)
test_data = remove_stopwords(test_data)

train_vector, test_vector = tf_idf_vectorizer(pd.concat([train_data, test_data]), train_data, test_data)

train_X, train_Y = oversample(train_vector, train_data['label'])

trained_classifier = train(linear_model.LogisticRegression(), train_X, train_Y)

predictions = trained_classifier.predict(test_vector)

test_data['label'] = predictions

predictions = np.array(predictions)

table = PrettyTable()
table.field_names = ['Design', 'General', 'Mean', 'Verdict']
design = (predictions == 1).sum()
general = (predictions == 0).sum()
mean = round(predictions.sum() / len(predictions), 2)
if mean >= 0.5:
    verdict = 'Design'
else:
    verdict = 'General'
table.add_row([design, general, mean, verdict])
print(table)

print('B R E A K D O W N')
print(test_data)


