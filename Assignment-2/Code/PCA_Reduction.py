from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#import os

#path = os.path.abspath("../Dataset")

def PCAReduction(extracted_feature):
    #extracted_feature = pd.read_csv(os.path.join(path, 'Feature_Extracted_Data.csv'), encoding='utf-8')
    #extracted_feature = pd.read_csv(os.path.join(path, 'CombinedData.csv'), encoding='utf-8')
    #label = extracted_feature['Label']
    
    #features = MinMaxScaler().fit_transform(extracted_feature.iloc[:, :-1])
    minmax = MinMaxScaler()
    features = minmax.fit_transform(extracted_feature)
    
    pc = PCA(0.95)
    pca = pc.fit(features)
    
    principal_components = pca.transform(features)
    
    columns = ['principal_components_'+str(i+1) for i in range(principal_components.shape[1])]
    
    final_pca = pd.DataFrame(principal_components,columns=columns)
    
    #final_df = pd.concat([final_pca, label], axis=1)
    final_df = final_pca.copy()
    
    #final_df.to_csv(os.path.join(path, "PCA_Reduced_Data.csv"), index=False, encoding='utf-8')
    #final_df.to_csv(os.path.join(path, "PCA_Reduced_Data_Original.csv"), index=False, encoding='utf-8')
    return pca, final_df, minmax
