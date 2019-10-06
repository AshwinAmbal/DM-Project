import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# extracted_feature = pd.read_csv("./../Extracted_Features.csv")
extracted_feature = pd.read_csv("/Users/avinash/Code/DM-Project/Assignment-1/Extracted_Features.csv")
extracted_feature.fillna(0)
features = extracted_feature.drop(extracted_feature.columns[0], axis=1).values

# Normalizing the features value to fit the PCA
features = StandardScaler().fit_transform(features)


pca = PCA(n_components=5)

principal_components = pca.fit_transform(features)

columns = ['principal_components_'+str(i+1) for i in range(5)]

final_pca = pd.Dataframe(data=principal_components, columns=columns)

final_pca.toCSV('./../principal_components_features.csv')
