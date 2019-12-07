import pandas as pd
import numpy as np
import Extract_Data
import os
import csv
from sklearn.metrics import classification_report
from ModelUtility import loadModel

def readCarbData(carbDirectory, testDataDirectory, path = os.path.abspath("../Dataset")):
    carbdirList = os.listdir(os.path.join(path, carbDirectory))
    data = []
    for item in carbdirList:
        with open(os.path.join(path, carbDirectory, item), encoding='utf-8') as fp:
            reader = csv.reader(fp)
            meal = []
            if(item.startswith('mealAmount')):
                meal.extend(row[0] for row in reader)
                with open(os.path.join(path, testDataDirectory, item[:4] + "Data" + item[-5:]), encoding='utf-8') as fp_train:
                    tempData = []
                    reader_train = csv.reader(fp_train)
                    tempData.extend(row[0] for row in reader_train)
                if(len(meal) >= len(tempData)):
                    print(len(tempData))
                    data.extend(meal[:len(tempData)])
                elif(len(meal) < len(tempData)):
                    data.extend(meal)
                    print([-1 for i in range(len(tempData) - len(meal))])
                    data.extend([-1 for i in range(len(tempData) - len(meal))])
    return data

def bucketList(carbData):
    bucketedData = []
    for row in carbData:
        row = int(row)
        if(row >= 0 and row < 10):
            bucketedData.append(0)
        elif(row >= 10 and row < 20):
            bucketedData.append(1)
        elif(row >= 20 and row < 30):
            bucketedData.append(2)
        elif(row >= 30 and row < 40):
            bucketedData.append(3)
        elif(row >= 40 and row < 50):
            bucketedData.append(4)
        elif(row >= 50 and row < 60):
            bucketedData.append(5)
        elif(row >= 60 and row < 70):
            bucketedData.append(6)
        elif(row >= 70 and row < 80):
            bucketedData.append(7)
        elif(row >= 80 and row < 90):
            bucketedData.append(8)
        elif(row >= 90 and row <= 100):
            bucketedData.append(9)
    return bucketedData

def findMapping(yhat, bucketedData):
    mapping = dict()
    count = dict()
    for (row1, row2) in zip(yhat, bucketedData):
        if(row1[0] not in count):
            count[row1[0]] = [0 for i in range(10)]
        count[row1[0]][row2] += 1
    print(count)
    for i in range(0, 10):
        mapping[i] = count[i].index(max(count[i]))
    return mapping

if __name__ == "__main__":
    _, df = Extract_Data.ExtractDataFeatures(train_dir=None, test_dir='RawTestData')
    #df, _ = Extract_Data.ExtractDataFeatures(train_dir="TrainRawData", test_dir=None)
    #carbData = readCarbData("CarbData", "TrainRawData")
    #bucketedData = bucketList(carbData)
    pca = loadModel('pca-weights')
    minmax = loadModel('minmax-weights')
    X_test = df.values
    X_test = minmax.transform(pd.DataFrame(X_test))
    X_test = pca.transform(pd.DataFrame(X_test))
    for cname in ['KMeans','DBScan']:
      print('Predicting the cluster assignment using ' +cname)
      clf = loadModel(cname)

      if cname == 'KMeans':
        yhat = clf.predict(X_test)
      else:
        yhat = clf.fit_predict(X_test)

      yhat = yhat.reshape(len(yhat), 1)
      pd.DataFrame(yhat).to_csv("result/"+cname+"-result.csv", index = False)
      print('Stored the result for '+cname+' clustering in the result folder')
      print()
      
      #mappedList = []
      #mapping = findMapping(yhat, bucketedData)
      #mappedList = [mapping[row[0]] for row in yhat]
      #print(classification_report(bucketedData, mappedList))