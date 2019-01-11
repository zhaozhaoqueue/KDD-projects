from __future__ import division
import itertools
import sys

import numpy as np
import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

np.set_printoptions(threshold=sys.maxsize)

#Data preprocessing
#Load data
raw_img = np.load("images.npy")
raw_lab = np.load("labels.npy")
#Reshape image data, from 6500*28*28 to 6500*748
img_data = raw_img.reshape(6500, -1)
#Encode label data
lab_data = to_categorical(raw_lab)
#Stratified sampling
train_validation_img, test_img, train_validation_lab, test_lab = train_test_split(img_data, lab_data, test_size=0.25, random_state=3, stratify=lab_data)
train_img, validation_img, train_lab, validation_lab = train_test_split(train_validation_img, train_validation_lab, test_size=0.2, random_state=4, stratify=train_validation_lab)

#Functions to be used
#Plot training accuracy and validation accuracy during each epoch
def plot_acc(hist):
    fig, ax = plt.subplots(1, 1)
    ax.plot(hist.history["acc"], color="b", label="Training Accuracy")
    ax.plot(hist.history["val_acc"], color="r", label="Validation Accuracy")
    legend = ax.legend(loc="best", shadow=True)
    plt.show()
#Compute and visualize confusion matrix
def plot_confusion_matrix(prediction,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function creates, prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """


    cm = confusion_matrix(np.argmax(test_lab, axis=1), np.argmax(prediction, axis=1))
    classes = list(range(10))



    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    print()
    print("Accuracy: ", accuracy_score(np.argmax(test_lab, axis=1), np.argmax(prediction, axis=1)))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
#Visualize image from pixel data
def show_pic(pixel_data):
    plt.imshow(pixel_data.reshape(28, 28), cmap="gray")
#Get list of incorrect prediction indexes
def wrong_list(prediction):
    y_pred = np.argmax(prediction, axis=1)
    y_true = np.argmax(test_lab, axis=1)
    wrong_index = []
    for i in range(len(y_pred)):
        if(y_pred[i] != y_true[i]):
            wrong_index.append(i)
    return wrong_index


#Create Model
model = Sequential() # declare model
model.add(Dense(100, input_shape=(28*28, ), kernel_initializer='random_normal')) # first layer
model.add(Activation('relu'))
#
model.add(Dense(100, activation="relu", kernel_initializer="random_normal", bias_initializer="random_normal"))
model.add(Dense(10, activation="relu", kernel_initializer="random_normal", bias_initializer="random_normal"))
#
model.add(Dense(10, kernel_initializer='random_normal')) # last layer
model.add(Activation('softmax'))


# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
history = model.fit(train_img, train_lab,
                    validation_data = (validation_img, validation_lab),
                    epochs=30,
                    batch_size=16)

# Model structure
# print(model.summary())

# Report Results
# print(history.history)

# Plot of training and validation accuracies
plot_acc(history)
print()

# Confusion matrix and its visualization on test dataset
prediction = model.predict(test_img)
plot_confusion_matrix(prediction)

print()

# Accuracy, recall, precision on test dataset
accuracy = accuracy_score(np.argmax(test_lab, axis=1), np.argmax(prediction, axis=1))
recall = recall_score(np.argmax(test_lab, axis=1), np.argmax(prediction, axis=1), labels=list(range(10)), average=None)
precision = precision_score(np.argmax(test_lab, axis=1), np.argmax(prediction, axis=1), labels=list(range(10)), average=None)
print("Accuracy: ", accuracy)
print("Recall: ", recall)
print("Precision: ", precision)
