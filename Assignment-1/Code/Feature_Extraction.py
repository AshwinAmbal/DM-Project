import pandas as pd
import numpy as np

import sys
sys.path.append("F:\3RDSEM\DM\Assignment_1\DM-Project\Assignment-1\Code")

from Utility import getDataFrame
fileNames = ["./../DataFolder/CGMSeriesLunchPat1.csv", "./../DataFolder/CGMSeriesLunchPat2.csv",
                   "./../DataFolder/CGMSeriesLunchPat3.csv", "./../DataFolder/CGMSeriesLunchPat4.csv",
                   "./../DataFolder/CGMSeriesLunchPat5.csv"]


def Feature_Extraction(df):
    feature_1_df = df.groupby(np.arange(len(df.columns))//6, axis=1).mean()
    
    feature_1_df.columns=['mean_'+str(i+1) for i, column in enumerate(feature_1_df.columns)]
    
    for i, columns in enumerate(feature_1_df.columns):
        feature_1_df['shifted_mean' + str(i+1)] = feature_1_df['mean_'+str(i+1)].shift(1)
    
    #==================================================================
    
    local_maxima = []
    for i in range(0,len(df.index)):
        indices = []
        for j in range(0, len(df.columns)-1): 
            if(df.iloc[i][df.columns[j]] >= df.iloc[i][df.columns[j-1]] and 
               df.iloc[i][df.columns[j]] > df.iloc[i][df.columns[j+1]]):
                indices.append(j)
        local_maxima.append(indices)
    
    local_minima = []
    for i in range(0,len(df.index)):
        indices = []
        for j in range(0, len(df.columns)-1): 
            if(df.iloc[i][df.columns[j]] <= df.iloc[i][df.columns[j-1]] and 
               df.iloc[i][df.columns[j]] < df.iloc[i][df.columns[j+1]]):
                indices.append(j)
        local_minima.append(indices)
    
    #==================================================================
        
    feature_2 = []
    for i,maxima in enumerate(local_maxima):
        global_maxima = 0
        temp_list = []
        for val in maxima:
            temp_list.extend(df.iloc[i][:].tolist())
            global_maxima = max(df.iloc[i][val], global_maxima)
        feature_2.append([global_maxima, (temp_list.index(global_maxima)) // 6 + 1 if temp_list != [] else -1])
    
    feature_2_df = pd.DataFrame(feature_2)
    feature_2_df.columns = ['Global_Maximum', 'Global_Maximum_Interval']
    
    #==================================================================
    
    segments = [(i) * 6 for i in range(len(df.columns)//6 + 1)]
    feature_3 = []
    for i, (maxima, minima) in enumerate(zip(local_maxima, local_minima)):
        count_local_maxima_interval = [0] * (len(df.columns)//6)
        count_local_minima_interval = [0] * (len(df.columns)//6)
        for val in maxima:
            for seg in range(1, len(segments)):
                if(val > segments[seg-1] and val <= segments[seg]):
                    count_local_maxima_interval[seg-1] += 1
        for val in minima:
            for seg in range(1, len(segments)):
                if(val > segments[seg-1] and val <= segments[seg]):
                    count_local_minima_interval[seg-1] += 1
        feature_3.append(count_local_maxima_interval + count_local_minima_interval)
    feature_3_df = pd.DataFrame(feature_3)
    feature_3_df.columns = ["Count_Local_Max_" + str(i) for i in range(1, len(segments))] + \
                            ["Count_Local_Min_" + str(i) for i in range(1, len(segments))]
    
    #==================================================================
    
    segments = [(i) * 6 for i in range(len(df.columns)//6 + 1)]
    feature_4 = []
    interval = -1
    for row, (maxima) in enumerate(local_maxima):
        diff_interval = [0] * (len(df.columns)//6)
        for val in maxima:
            for seg in range(1, len(segments)):
                if(val > segments[seg-1] and val <= segments[seg]):
                    interval = seg-1
                    break
            local_maxima_interval = df.iloc[row][val]
            prev = val - 1
            prev_local_minimum = 1000
            while(prev > segments[interval]):
                prev_local_minimum = min(df.iloc[row][prev], prev_local_minimum)
                prev -= 1
            prev_local_minimum = min(df.iloc[row][prev], prev_local_minimum)
            prev_local_minimum %= 1000
            diff = local_maxima_interval - prev_local_minimum
            diff_interval[interval] = diff
        feature_4.append(diff_interval)
    
    feature_4_df = pd.DataFrame(feature_4)            
    feature_4_df.columns = ["Diff_Local_Max_Min_Interval_" + str(i) for i in range(1, len(segments))]
    
    #==================================================================

    segments = [(i) * 6 for i in range(len(df.columns) // 6 + 1)]
    feature_5 = {}
    for i in range(len(segments) - 1):
        df1 = df.iloc[:, segments[i]:segments[i + 1]]
        diff1 = df1[df1.columns[::-1]].diff(axis=1)
        if 'cgmSeries_30' in diff1.columns:
            diff1['cgmSeries_30'].fillna(0, inplace=True)
        sum1 = diff1.sum(axis=1)
        feature_5[i] = sum1

    feature_5_df = pd.DataFrame.from_dict(feature_5)
    feature_5_df.columns = ['CGM_Displacement_Interval_' + str(i) for i in range(1, len(segments))]

    #==================================================================
    segments = [(i) * 6 for i in range(len(df.columns) // 6 + 1)]
    feature_6 = {}
    for i in range(len(segments) - 1):
        df1 = df.iloc[:, segments[i]:segments[i + 1]]
        diff1 = df1[df1.columns[::-1]].diff(axis=1)
        if 'cgmSeries_30' in diff1.columns:
            diff1['cgmSeries_30'].fillna(0, inplace=True)
        mean1 = diff1.mean(axis=1)
        feature_6[i] = mean1

    feature_6_df = pd.DataFrame.from_dict(feature_6)
    feature_6_df.columns = ['CGM_Velocity_Interval_' + str(i) for i in range(1, len(segments))]
    #==================================================================
    final_df = pd.concat([df, feature_1_df, feature_2_df, feature_3_df, feature_4_df, feature_5_df, feature_6_df], axis=1)
    return final_df

final_df = pd.DataFrame()
for fileName in fileNames:
    df = getDataFrame(fileName)
    returned_df = Feature_Extraction(df)
    final_df = pd.concat([final_df, returned_df], axis=0, ignore_index=True)

final_df.to_csv("./../Extracted_Features.csv")
