import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/AshwinAmbal/Desktop/MS ASU/Fall'19/DM/Project-1/DataFolder/CGMSeriesLunchPat4.csv")

columns = []
for column in df.columns:
    columns.append(column.replace(' ', ''))

df.columns = columns

df = df[['cgmSeries_' + str(i) for i in range(1, 31)]].copy()

temp = df.copy()

for i in range(1, 31):
    if i == len(df.columns):
        df['cgmSeries_' + str(i)] = df.apply(
        lambda row: (row['cgmSeries_' + str(i - 1)]) 
        if np.isnan(row['cgmSeries_' + str(i)]) else row['cgmSeries_' + str(i)],
        axis=1
        ) 
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
        
df = df.fillna(method='bfill', axis=1)
df = df.fillna(method='ffill', axis=1)

mean = df.groupby(np.arange(len(df.columns))//6, axis=1).mean()
        
mean.columns=['mean_'+str(i+1) for i, column in enumerate(mean.columns)]

for i, columns in enumerate(mean.columns):
    mean['shifted_mean' + str(i+1)] = mean['mean_'+str(i+1)].shift(1)

final_df = pd.concat([df,mean], axis=0)    

transposed_df = df.T
