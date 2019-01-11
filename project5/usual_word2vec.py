import random
import re
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from scipy.spatial.distance import cosine
# from keras.utils.np_utils import to_categorical
# from keras.models import Sequential
# from keras.layers import Dense, Activation
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import classification_report_imbalanced
from gensim.models import Word2Vec, KeyedVectors


# Preprocessing functions
# remove punctuations and numbers, lowercase and stem words, remove stopwords
def preprocessing(text):
    # text = text.lower()
    # #remove stopwords
    # sw = stopwords.words("english")
    # text = " ".join([word for word in re.split(r'[\s,.?]+', text) if word not in sw])
    # #remove multiple spaces
    # text = re.sub(r'\s+', ' ', text)
    # #remove numbers
    # text = re.sub(r'\d', '', text)
    #remove punctuations except "&"
    text = re.sub('/-', ' ', text)
    text = re.sub('[^\w\s&]', '', text)
    text = re.sub('&', ' & ')  # split "&" since googlenews embeddings contains "&"
    # clearn numbers
    text = re.sub('[0-9]{5,}', '#####', text)
    text = re.sub('[0-9]{4}', '####', text)
    text = re.sub('[0-9]{3}', '###', text)
    text = re.sub('[0-9]{2}', '##', text)
    # #stem words
    # stemmer = SnowballStemmer("english")
    # text = " ".join([stemmer.stem(word) for word in text.split()])
    # #try lemmatize words
    # lemmatizer = WordNetLemmatizer()
    #text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text
# reference: https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings


# Word2Vec
# load glove pretrained word2vector
def glove_load(fname):
    model = {}
    f = open(fname, "r")
    for line in f:
        wv_ls = line.split()
        word = "".join(wv_ls[:-300])
        embeddings = np.array(wv_ls[-300:], dtype="float32")
        model[word] = embeddings
    return model

# average
def avg_sentence(text, w2v):
    v = np.zeros(300)
    wl = text.split()
    wl = [word for word in wl if word in w2v]
    for word in wl:
        v += w2v[word]
    if(len(wl) == 0):
        return np.zeros(300)
    return v/len(wl)

# lowest similarity
def low_sentence(text, w2v):
    word_ls = text.split()
    word_ls = [word for word in word_ls if word in w2v]
    sim_ls = np.zeros(len(word_ls))
    for i in range(len(word_ls)):
        for word in word_ls:
            sim_ls[i] += 1 - cosine(w2v[word_ls[i]], w2v[word]) #cosine similarity of each 2 words
    lowest = np.argmin(sim_ls)
    return w2v[word_ls[lowest]]


# Load train data and test data
train_data = pd.read_csv("train1.csv")
test_data = pd.read_csv("val.csv")

# Preprocess train data
train_data["question_text"] = train_data["question_text"].apply(preprocessing)

# Word2Vec embedding
#load google news pretrained word2vector
w2v = KeyedVectors.load_word2vec_format("./embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin", binary=True)
## load glove pretrained word2vector
#w2v = glove_load("./embeddings/glove.840B.300d/glove.840B.300d.txt")

X_train = np.zeros((train_data.shape[0], 300))
for index, row in train_data.iterrows():
    # #average vector
    # X_train[index, :] = avg_sentence(row["question_text"], w2v).reshape(1, -1)
    #lowest sim vec
    X_train[index, :] = low_sentence(row["question_text"], w2v).reshape(1, -1)
y_train = train_data["target"]
# Convert test data to vector
X_test = np.zeros((test_data.shape[0], 300))
for index, row in test_data.iterrows():
    # #average vector
    # X_test[index, :] = avg_sentence(row["question_text"], w2v).reshape(1, -1)
    #lowest sim vec
    X_test[index, :] = low_sentence(row["question_text"], w2v).reshape(1, -1)
y_test = test_data["target"]


# Resample
ros = RandomOverSampler(random_state=1)
X_train, y_train = ros.fit_resample(X_train, y_train)
print("train X shape after resample: ", X_train.shape)
print("train y shape after resample: ", y_train.shape)

# Logistic Regression with 10-fold cross validation
lg_clf = LogisticRegressionCV(cv=10, scoring="f1", solver="sag", random_state=1)
lg_clf.fit(X_train, y_train)
print("Logistic Regression result\n")
# print("training f1_score: ", lg_clf.scores_)
lg_pred = lg_clf.predict(X_test)
print("confusion matrix:",confusion_matrix(y_test, lg_pred, labels=[1, 0]).T)
print()
print(classification_report_imbalanced(y_test, lg_pred))
print("f1 score: ", f1_score(y_test, lg_pred))
print()

# SVM
svm_clf = SVC(random_state=1)
svm_clf.fit(X_train, y_train)
print("SVM result\n")
svm_pred = svm_clf.predict(X_test)
print("confusion matrix: ", confusion_matrix(y_test, svm_pred, labels=[1, 0]).T)
print()
print(classification_report_imbalanced(y_test, svm_pred))
print("f1 score: ", f1_score(y_test, svm_pred))
print()

# Random Forest
rf_clf = RandomForestClassifier(criterion="entropy", max_depth=15, random_state=1)
rf_clf.fit(X_train, y_train)
print("Random forest result\n")
rf_pred = rf_clf.predict(X_test)
print("confusion matrix: ", confusion_matrix(y_test, rf_pred, labels=[1, 0]).T)
print()
print(classification_report_imbalanced(y_test, rf_pred))
print("f1 score: ", f1_score(y_test, rf_pred))
