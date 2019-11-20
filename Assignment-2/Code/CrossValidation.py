import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier

"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
"""
import Extract_Data
import PCA_Reduction

random.seed(42)
np.random.seed(42)
path = os.path.abspath("../Dataset")

#df = pd.read_csv(os.path.join(path, 'CombinedData.csv'), encoding='utf-8')
#df = pd.read_csv(os.path.join(path, 'Feature_Extracted_Data.csv'), encoding='utf-8')
#df = pd.read_csv(os.path.join(path, 'PCA_Reduced_Data.csv'), encoding='utf-8')
#df = pd.read_csv(os.path.join(path, 'PCA_Reduced_Data_Original.csv'), encoding='utf-8')

def getClassifier(cname, prob=False):
    if cname == 'LogisticRegression':
        return LogisticRegression(random_state=42, solver='liblinear')
    elif cname == 'RandomForest':
        return RandomForestRegressor(n_estimators=20, random_state=0)
    elif cname == 'SVM':
        return SVC(kernel='linear', probability=prob)
    elif cname == 'DecisionTreeClassifier':
        return DecisionTreeClassifier(random_state=42)
    elif cname == 'KNeighborsClassifier':
        return KNeighborsClassifier(n_neighbors=5)
    elif cname == 'GaussianNB':
        return GaussianNB()
    elif cname == 'VotingClassifier':
        classifiers = ['LogisticRegression', 'SVM', 'KNeighborsClassifier']
        fitted_classifiers = []
        for name in classifiers:
            fitted_classifiers.append(tuple([name, getClassifier(name, True)]))
        return VotingClassifier(estimators=fitted_classifiers, voting='soft')
        


def CrossValidation(df,cname, n_splits=2):
    X = df.iloc[:, :-1].values
    y = df['Label'].values
    skf = StratifiedKFold(n_splits=n_splits)
    skf.get_n_splits(X, y)

    all_test_labels = []
    all_pred_labels = []
    for i, (train_index, test_index) in enumerate(skf.split(X,y)):
        #print("\n\n\nRUNNING FOLD: {}".format(i))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        pca, X_train_df, minmax = PCA_Reduction.PCAReduction(pd.DataFrame(X_train))
        X_train = X_train_df.values
        X_test = minmax.transform(pd.DataFrame(X_test))
        X_test = pca.transform(pd.DataFrame(X_test))
        clf = getClassifier(cname).fit(X_train, y_train)
        yhat = clf.predict(X_test)

        if cname == 'RandomForest':
          yhat = list(map(lambda x: int(round(x)), clf.predict(X_test)))

        #gnb = GaussianNB()
        #yhat = gnb.fit(X_train, y_train).predict(X_test)

        #clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
        #yhat = clf.predict(X_test)

        #clf = KNeighborsClassifier(n_neighbors=5)
        #clf.fit(X_train, y_train)
        #yhat = clf.predict(X_test)

        """
        X_train = np.reshape(X_train, (len(X_train), X.shape[1], 1))
        X_test = np.reshape(X_test, (len(X_test), X.shape[1], 1))

        model = Sequential()
        #model.add(Dense(int(X.shape[1]), input_shape=(X.shape[1],), activation='relu'))
        model.add(LSTM(X.shape[1], input_shape=(X.shape[1],1)))
        #model.add(Dropout(0.4))
        model.add(Dense(int(X.shape[1]/2), activation='relu'))
        #model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=25, verbose=0)
        yhat = model.predict_classes(X_test, verbose=0)
        """
        #print(classification_report(y_test, yhat))
        all_test_labels.extend(y_test)
        all_pred_labels.extend(yhat)
    print("\n\nOverall Classification Report of "+ cname +":\n")
    print(classification_report(all_test_labels, all_pred_labels))
    return clf


def main():
    train_df, test_df = Extract_Data.ExtractDataFeatures(train_dir='TrainRawData', test_dir=None)
    #temp_df = train_df.copy()
    #df = pd.concat([temp_df.loc[:,'mean_1':'Global_Maximum'], temp_df['Label']], axis=1)
    classifiers = ['LogisticRegression', 'RandomForest', 'SVM', 'DecisionTreeClassifier', 'KNeighborsClassifier', 'GaussianNB', 'VotingClassifier']
    for cname in classifiers:
      CrossValidation(train_df, cname, 10)
    


if __name__ == '__main__':
    main()
