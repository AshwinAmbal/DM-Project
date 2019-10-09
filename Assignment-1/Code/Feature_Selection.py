import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


extracted_feature = pd.read_csv("/Users/avinash/Code/DM-Project/Assignment-1/Extracted_Features.csv")

for i in range(36):
  extracted_feature = extracted_feature.drop(extracted_feature.columns[0], axis=1)

# print(extracted_feature.columns)
features = extracted_feature.fillna(0)
print('The shape of Feature Matrix: ', features.shape)
# Normalizing the features value to fit the PCA
features = MinMaxScaler().fit_transform(features)

# Creating a PCA Class to select top 5 components
pca = PCA(n_components=5)

# Executing the PCA calcualtion
principal_components = pca.fit_transform(features)

columns = ['principal_components_'+str(i+1) for i in range(5)]

# Creating the Matrix of shape (5 x 33)
# Rows: Principal Components selected by the PCA
# Columns: Features contribution to each PC

final_pca = pd.DataFrame(pca.components_,columns=extracted_feature.columns,index = ['PC-1','PC-2', 'PC-3', 'PC-4', 'PC-5'])

# Finding the top 5 components
print(final_pca.abs().sum().nlargest(5))


# Storing the final Principal Component Matrix
final_pca.to_csv('./../pc_matrix.csv')
