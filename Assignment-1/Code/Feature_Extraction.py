import pandas as pd
import numpy as np

from Utility import getDataFrame

df = getDataFrame("./../DataFolder/CGMSeriesLunchPat1.csv")

mean = df.groupby(np.arange(len(df.columns))//6, axis=1).mean()

mean.columns=['mean_'+str(i+1) for i, column in enumerate(mean.columns)]

for i, columns in enumerate(mean.columns):
    mean['shifted_mean' + str(i+1)] = mean['mean_'+str(i+1)].shift(1)

final_df = pd.concat([df,mean], axis=0)

transposed_df = df.T
