# Team Members

 - Avinash Sivaraman (1215054529 - asivara6@asu.edu)
 - Kaviya Kalaichelvan(121512889 - kkalaich@asu.edu)
 - Ashwin Karthik Ambalavanan (1215105307 - aambalav@asu.edu)
 - Aishwarya Sankararaman(125102941- asanka17@asu.edu)

# Data Mining Assignment 3

This assignment is to cluster the time series based glucose level data. KMeans algorithm is chosen for clustering and the value of K is set as 10. SSE is calculated in the form of inertia or within-cluster sum-of-squares calculation. Clustering is also performed using DBSCAN algorithm with the epsilon value set to 0.1 and minimum number of samples set to 1. Also the meal amount data for five patients is put into buckets based on ranges to map the buckets and cluster assignments obtained from each of the clustering algorithm and compare with the results.

## Testing the assignment

  1. Place the test file in the Dataset > RawTestData Folder
  2. As the model already trained and stored as the pickle file, Please navigate
     inside the Code folder and run the `Predict.py`
     ```bash
        > cd Code
        > python3 Predict.py
     ```
  3. Once the `Predict.py` runs successfully, Please navigate to the result folder
   in the same directory (Code) to view the results of each classifier.

    +-- Dataset
    |   +-- RawTestData
    |   |  +-- Place your test file
    +-- Code
    |   +-- Predict.py
    |   +-- result
        |   +-- Kmeans-result.csv
        |   +-- DBScan-result.csv

  ```
    Note: The result files will be generated only when you run the Predict.py
  ```
## Files and it's usage

`Extract_Data.py` - Appends the meal/nomeal labels i.e 1/0 for the corresponding
                  files of all 5 patients. Cleans the data by replacing NaN values
                  with mean from previous data. It then calls Feature_Extraction
                  function.

`Feature_Extraction.py` - Calling the API creates the following features -CGM velocity,
                        CGM displacement, count of local maxima and minima per interval
                        and shifted mean.

`PCA_Reduction.py` -  Reduces the data dimensions of data to 3 features after recognizing
                    largely varying features and transforming them.

`ModelUtility.py` -  It saves the Model as the pickle file and it loads it up while
                   testing.

`Cluster.py` -  It fits the data with the Kmeans and DBScan clustering and saves the trained cluster assignments
            as a pickle file in the Model directory. 

`Predict.py` -  It contains the function call to run the Kmeans and DBSCAN clustering. It also holds the bucketing algorithm for the carbs data which can be used for comparison.
              

