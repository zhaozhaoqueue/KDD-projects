import random
import re
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
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



# Load train data and test data
train_data = pd.read_csv("train1.csv")
test_data = pd.read_csv("val.csv")

# Preprocess train data
train_data["question_text"] = train_data["question_text"].apply(preprocessing)

# Vectorize text
# TF-TDF
tfidf = TfidfVectorizer(ngram_range=(1,1), max_df=0.5, min_df=10)
X_train = tfidf.fit_transform(train_data["question_text"])#.toarray()
print(tfidf.get_feature_names())
print()
#X_train = pd.DataFrame(X_train, columns=tfidf.get_feature_names())
# y_train = np.array(train_data["target"]).reshape(-1, 1)
y_train = train_data["target"]
print("vectorized training X shape: ", X_train.shape)
print("vectorized training y shape: ", y_train.shape)
# Convert test data to vector
X_test = tfidf.transform(test_data["question_text"])#.toarray()
# X_test = pd.DataFrame(X_test, columns=tfidf.get_feature_names())
y_test = test_data["target"]
print("vectorized test X shape: ", X_test.shape)
print("vectorized test y shape: ", y_test.shape)

# Resample
ros = RandomOverSampler(random_state=1)
X_train, y_train = ros.fit_resample(X_train, y_train)
print("train X shape after resample: ", X_train.shape)
print("train y shape after resample: ", y_train.shape)

# Logistic Regression with 10-fold cross validation
lg_clf = LogisticRegressionCV(cv=10, scoring="f1", solver="sag", random_state=1)
lg_clf.fit(X_train, y_train)
print("Logistic Regression result\n")
#print("training f1_score: ", lg_clf.scores_)
lg_pred = lg_clf.predict(X_test)
print("confusion matrix:",confusion_matrix(y_test, lg_pred, labels=[1, 0]).T)
print(classification_report_imbalanced(y_test, lg_pred))
print("f1 score: ", f1_score(y_test, lg_pred))
