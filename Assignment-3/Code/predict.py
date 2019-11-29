import pandas as pd
import numpy as np
import Extract_Data
from ModelUtility import loadModel

if __name__ == "__main__":
    _, df = Extract_Data.ExtractDataFeatures(train_dir=None, test_dir='RawTestData')
    pca = loadModel('pca-weights')
    minmax = loadModel('minmax-weights')
    X_test = df.values
    X_test = minmax.transform(pd.DataFrame(X_test))
    X_test = pca.transform(pd.DataFrame(X_test))
    print()
    for cname in ['KMeans',]:
      print('Predicting the cluster assignment using' +cname)
      clf = loadModel(cname)
      yhat = clf.predict(X_test)

      yhat = yhat.reshape(len(yhat), 1)
      pd.DataFrame(yhat).to_csv("result/"+cname+"-result.csv")
      print('Stored the result for '+cname+' clustering in the result folder')
      print()