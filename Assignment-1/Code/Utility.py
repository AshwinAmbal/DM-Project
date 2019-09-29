import pandas as pd
import numpy as np

def getDataFrame(fileName):
  df = pd.read_csv(fileName)

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
  return df
