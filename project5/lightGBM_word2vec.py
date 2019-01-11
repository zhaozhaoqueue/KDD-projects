import random
import re
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import classification_report_imbalanced
from gensim.models import Word2Vec, KeyedVectors
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegressionCV
from scipy.spatial.distance import cosine
# from keras.utils.np_utils import to_categorical
# from keras.models import Sequential
# from keras.layers import Dense, Activation
import lightgbm as lgb


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
    if(not word_ls):
        return np.zeros(300)
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
# load googlenews pretrained word2vector
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
print("test X shape: ", X_test.shape)
print("test y shape: ", y_test.shape)
print("train X shape before resample: ", X_train.shape)
print("train y shape before resample: ", X_train.shape)
print("\n")

#

# lightGBM without resampling
print("LightGBM without resampling begins\n")

# Create validation data
lgb_x_train, lgb_x_val,lgb_y_train, lgb_y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=1)

# Construct training and validation datasets for lightGBM
lgb_train = lgb.Dataset(lgb_x_train, lgb_y_train)
lgb_val = lgb.Dataset(lgb_x_val, lgb_y_val, reference=lgb_train)

# List of parameters
params = {
        "objective_type": "binary",
        "boosting_type": "gbdt",
        "metric": ["auc", "binary_logloss"],
        "num_leaves": 90,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "max_depth": 10,
        "seed": 1,
        "unbalance": "true"}
        #"scale_pos_weight":99}
# Train
lgb_clf = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=[lgb_train, lgb_val],
                early_stopping_rounds=10)

# Predict
lgb_pred_prob = lgb_clf.predict(X_test, num_iteration=lgb_clf.best_iteration)
lgb_pred = np.int32(lgb_pred_prob>=0.5)

print("confusion matrix: ", confusion_matrix(y_test, lgb_pred, labels=[1, 0]).T)
print()
print(classification_report_imbalanced(y_test, lgb_pred))
print("f1 score: ", f1_score(y_test, lgb_pred))

print("LightGBM without resampling ends\n")


# Resample
ros = RandomOverSampler(random_state=1)
X_train, y_train = ros.fit_resample(X_train, y_train)
print("train X shape after resample: ", X_train.shape)
print("train y shape after resample: ", y_train.shape)

# lightGBM with resampling
print("LightGBM begins")

# Create validation data
lgb_x_train, lgb_x_val,lgb_y_train, lgb_y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=1)

# Construct training and validation datasets for lightGBM
lgb_train = lgb.Dataset(lgb_x_train, lgb_y_train)
lgb_val = lgb.Dataset(lgb_x_val, lgb_y_val, reference=lgb_train)

# List of parameters
params_1 = {
        "objective_type": "binary",
        "boosting_type": "gbdt",
        "metric": ["auc", "binary_logloss"],
        "num_leaves": 90,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "max_depth": 10,
        "seed": 1}

# Train
lgb_clf_1 = lgb.train(
                    params_1,
                    lgb_train,
                    num_boost_round=20,
                    valid_sets=lgb_val,
                    early_stopping_rounds=5)

# Predict
lgb_pred_prob_1 = lgb_clf_1.predict(X_test, num_iteration=lgb_clf_1.best_iteration)
lgb_pred_1 = np.int32(lgb_pred_prob_1>=0.5)

print("confusion matrix: ", confusion_matrix(y_test, lgb_pred_1, labels=[1, 0]).T)
print(classification_report_imbalanced(y_test, lgb_pred_1))
print("f1 score: ", f1_score(y_test, lgb_pred_1))

print("LightGBM ends")
