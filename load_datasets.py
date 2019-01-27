import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import *

def load_abalone():

    data = pd.read_csv('data/abalone.csv', header=None)
    data = np.array(data.dropna())

    X = data[:,1:-1]
    y = data[:,-1]

    gender = pd.read_csv('data/abalone.csv', header=None, usecols=[0])
    le = preprocessing.LabelEncoder()
    gender = gender.apply(le.fit_transform)
    enc = preprocessing.OneHotEncoder()
    enc.fit(gender)
    gender = enc.transform(gender).toarray()

    X = np.append(gender, X, 1).astype('float')

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def load_letters():
    data = pd.read_csv('data/letter.csv')
    data = np.array(data.dropna())
    X = data[:,1:].astype('float')
    y = data[:,0]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


