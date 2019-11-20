import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

import Extract_Data
import PCA_Reduction
from ModelUtility import saveModel

random.seed(42)
np.random.seed(42)
path = os.path.abspath("../Dataset")


def getClassifier(cname):
  if cname == 'LogisticRegression':
    return LogisticRegression(random_state=42, solver='liblinear')
  elif cname == 'RandomForest':
    return RandomForestRegressor(n_estimators=20, random_state=0)
  elif cname == 'SVM':
    return SVC(kernel='linear')
  elif cname == 'DecisionTreeClassifier':
    return DecisionTreeClassifier(random_state=42)
  elif cname == 'KNeighborsClassifier':
    return KNeighborsClassifier(n_neighbors=2)
  elif cname == 'GaussianNB':
        return GaussianNB()

def train(X, y, cname):
  clf = getClassifier(cname).fit(X, y)
  saveModel(clf, cname)

if __name__ == '__main__':
  df, _ = Extract_Data.ExtractDataFeatures(train_dir='TrainRawData', test_dir=None)
  X = df.iloc[:, :-1].values
  y = df['Label'].values

  pca, X_train_df, minmax = PCA_Reduction.PCAReduction(pd.DataFrame(X))
  X = X_train_df.values
  for cname in ['LogisticRegression', 'RandomForest', 'SVM', 'DecisionTreeClassifier']:
      train(X, y, cname)
  saveModel(pca, 'pca-weights')
  saveModel(minmax, 'minmax-weights')
