# DM-Project
Data Mining CGM Data Processing
Data Mining Assignment 3
This assignment is to cluster the time series based glucose level data. KMeans algorithm is chosen for clustering and the value of K is set as 10. SSE is calculated in the form of inertia or within-cluster sum-of-squares calculation.

Testing the assignment
Place the test file in the Dataset > RawTestData Folder
As the model already trained and stored as the pickle file, Please navigate inside the Code folder and run the Predict.py
   > cd Code
   > python3 Predict.py
Once the Predict.py runs successfully, Please navigate to the result folder in the same directory (Code) to view the result of clustering.
+-- Dataset
|   +-- RawTestData
|   |  +-- Place your test file
+-- Code
|   +-- Predict.py
|   +-- result
    |   +-- KMeans-result.csv
 
  Note: The result file will be generated only when you run the Predict.py
Files and it's usage
Extract_Data.py - Appends the meal/nomeal labels i.e 1/0 for the corresponding files of all 5 patients. Cleans the data by replacing NaN values with mean from previous data. It then calls Feature_Extraction function.

Feature_Extraction.py - Calling the API creates the following features -CGM velocity, CGM displacement, count of local maxima and minima per interval and shifted mean.

PCA_Reduction.py - Reduces the data dimensions of data to 3 features after recognizing largely varying features and transforming them.

ModelUtility.py - It saves the Model as the pickle file and it loads it up while testing.

Train.py - It trains the data with KMeans clustering algorithm and saves the trained cluster assignment as a pickle file in the Model directory.

Predict.py - It contains the function call to run KMeans clustering.
