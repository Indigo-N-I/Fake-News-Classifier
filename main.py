# followed DataFlair's tutorial
import numpy as np
import pandas as pd
import itertools

import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
# add more features later
from sklearn.metrics import accuracy_score, confusion_matrix

DISPLAY = False

cwd = os.getcwd()
path = os.path.join(cwd, "news")

df = pd.read_csv(os.path.join(path, "news.csv"))
labels = df.label

if DISPLAY:
    # check info was gotten
    print(df.shape)
    print(df.head())
    print(labels.head())


x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size = .3, random_state = 1)

# Use TfidfVectorizor
tfid_vect = TfidfVectorizer(stop_words = 'english', max_df = .7)
x_train = tfid_vect.fit_transform(x_train)
x_test = tfid_vect.transform(x_test)


pac = PassiveAggressiveClassifier(max_iter = 50)
# print(x_train.shape, x_test.shape)
pac.fit(x_train, y_train)

y_pred = pac.predict(x_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100, 2)}')


con_mat = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
print(con_mat)
