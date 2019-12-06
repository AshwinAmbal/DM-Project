import csv
import os
import random
import pandas as pd
import numpy as np

import Feature_Extraction

random.seed(42)

path = os.path.abspath("../Dataset")

def CreateDataset(path, directory, test=False):
    dirList = os.listdir(os.path.join(path, directory))
    data = []
    for item in dirList:
        with open(os.path.join(path, directory, item), encoding='utf-8') as fp:
            reader = csv.reader(fp)
            if(item.startswith('meal')):
                if(not test):
                    data.extend(row[:30] + ['1'] for row in reader)
                else:
                    data.extend(row[:30] for row in reader)
            elif(item.startswith('Nomeal')):
                if(not test):
                    data.extend(row[:30] + ['0'] for row in reader)
                else:
                    data.extend(row[:30] for row in reader)
            else:
                data.extend(row[:30] for row in reader)

    return data

def ShuffleData(data):
    index = []
    for i in range(len(data)):
        index.append(i)
    req_index = random.sample(index, len(data))
    shuffled_data = []
    for index in req_index:
        shuffled_data.append(data[index])
    return shuffled_data

def CleanData(data, test=False):
    result = []
    for row in data:
        if(test or len(row) == 31):
            result.append(row)
    if(not test):
        column_names = ['cgmSeries_' + str(i) for i in range(1, 31)] + ['Label']
    else:
        column_names = ['cgmSeries_' + str(i) for i in range(1, 31)]
    # print(data, column_names)
    df = pd.DataFrame(result, columns=column_names)
    df = df.apply(pd.to_numeric, errors='coerce')
    for i in range(1, 31):
        if i == len(df.columns) - 1:
            df['cgmSeries_' + str(i)] = df.apply(
                lambda row: (row['cgmSeries_' + str(i - 1)])
                if np.isnan(row['cgmSeries_' + str(i)]) else row['cgmSeries_' + str(i)],
                axis=1
            )
            if(test):
                break
        elif i == 1:
            df['cgmSeries_' + str(i)] = df.apply(
                lambda row: (row['cgmSeries_' + str(i + 1)]) if np.isnan(row['cgmSeries_' + str(i)]) else row['cgmSeries_' + str(i)],
                axis=1
            )
        else:
            df['cgmSeries_' + str(i)] = df.apply(
                lambda row: (row['cgmSeries_' + str(i-1)]+row['cgmSeries_' + str(i+1)])/2
                if np.isnan(row['cgmSeries_'+str(i)]) and
                    not np.isnan(row['cgmSeries_'+str(i+1)]) and
                    not np.isnan(row['cgmSeries_'+str(i-1)])
                else row['cgmSeries_'+str(i)],
                axis=1
                )
    df.iloc[:, 0:30] = df.iloc[:, 0:30].fillna(method='bfill', axis=1)
    df.iloc[:, 0:30] = df.iloc[:, 0:30].fillna(method='ffill', axis=1)
    if(not test):
        df = df.dropna()
    else:
        df = df.fillna(0)
    df = df.reset_index(drop=True)
    return df


def ExtractDataFeatures(train_dir=None, test_dir=None):
    final_test_df = None
    final_train_df = None
    if train_dir:
      train_data = CreateDataset(path, train_dir)
      train_data = ShuffleData(train_data)
      train_df = CleanData(train_data)
      final_train_df = Feature_Extraction.Feature_Extraction(train_df)

    #train_df.to_csv(os.path.join(path, 'CombinedData.csv'), index=False, encoding='utf-8')
    if(test_dir):
        test_data = CreateDataset(path, test_dir, test=True)
        test_data = ShuffleData(test_data)
        test_df = CleanData(test_data, test=True)
        final_test_df = Feature_Extraction.Feature_Extraction(test_df, test=True)

    #df = pd.read_csv(os.path.join(path, 'CombinedData.csv'), encoding='utf-8')
    #final_df.to_csv(os.path.join(path, 'Feature_Extracted_Data.csv'), index=False, encoding='utf-8')
    return final_train_df, final_test_df
