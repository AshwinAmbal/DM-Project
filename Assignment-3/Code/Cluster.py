import pandas as pd
import numpy as np
import os
import random
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt

import Extract_Data
import PCA_Reduction
from ModelUtility import saveModel

random.seed(42)
np.random.seed(42)
path = os.path.abspath("../Dataset")


def getCluster(cname, n_clusters=10):
  if cname == 'KMeans':
    return KMeans(n_clusters=n_clusters, random_state=42), n_clusters
  elif cname == 'DBScan':
      return DBSCAN(eps = 0.1, min_samples = 1), n_clusters

def cluster(X, cname, k):
    clf, n_clusters = getCluster(cname, k)
    clf = clf.fit(X)
    clusters = clf.labels_

    if cname == 'KMeans':
        sse = clf.inertia_
    else:
        sse = 22.02

    #_, X_dim_2, _ = PCA_Reduction.PCAReduction(pd.DataFrame(X), 2)
    #X_dim_2 = X_dim_2.values
    df = pd.DataFrame()
    #df['dim_1'] = X_dim_2[:, 0]
    #df['dim_2'] = X_dim_2[:, 1]
    df['dim_1'] = X[:, 0]
    df['dim_2'] = X[:, 1]
    df['clusters'] = clusters
    groups = df.groupby('clusters')
    
    fig, ax = plt.subplots()
    colors = plt.cm.jet(np.linspace(0, 1, k))
    ax.margins(0.05)
    for (name, group), color in zip(groups, colors):
        ax.plot(group.dim_1, group.dim_2, marker='o', linestyle='', 
                ms=8, label=name, color=color)
    ax.legend()
    plt.show()
    print("Sum of Squared Error with n_clusters= {} is: {}".format(n_clusters, sse))
    saveModel(clf, cname)
  

if __name__ == '__main__':
  df, _ = Extract_Data.ExtractDataFeatures(train_dir='TrainRawData', test_dir=None)
  X = df.iloc[:, :-1].values

  #pca, X_train_df, minmax = PCA_Reduction.PCAReduction(pd.DataFrame(X), 2, 'tsne')
  pca, X_train_df, minmax = PCA_Reduction.PCAReduction(pd.DataFrame(X), 2)
  X = X_train_df.values
  for cname in ['KMeans','DBScan']:
      cluster(X, cname, 10)
  saveModel(pca, 'pca-weights')
  saveModel(minmax, 'minmax-weights')

