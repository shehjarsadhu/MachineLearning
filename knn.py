import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree 
from collections import Counter
class knn():
    def fit(self,X_train_np,y_train_np):
        self.X_train_np = X_train_np
        self.y_train_np = y_train_np
    #....returns a list of predicted values....
    def predict(self,X_train, X_valid_np,y_train_np,dist,k):
        tree = KDTree(X_train, leaf_size=2,metric=dist)
        #..for float conversions..
        X_copy_test =[]
        for i in range(len(X_valid_np)):
            X_copy_test.append(list(map(float,X_valid_np[:len(X_valid_np)][i])))

        dist, ind = tree.query(X_copy_test[:len(X_copy_test)],k)
        predictions = []
        for i in range(len(X_valid_np)):
            freq_counts =  Counter(y_train_np[ind[i]])
            max_oucc_chars = max(freq_counts, key=freq_counts.get)
            predictions.append(max_oucc_chars)
        for i in range(len(predictions)):
            print(predictions[i])
        #print("length of my test set: ",len(X_valid_np))
        #print("length of my output: ", len(predictions))
        return predictions
def main():
    train = sys.argv[1]
    test = sys.argv[2]
    f_name = sys.argv[3] #json file
    #....for json config files.....#
    with open(f_name, 'r') as f:
        df_json = json.load(f)
    #reading the dataset 
    df = pd.read_csv(train, header=None)
        
    #drop by column.dropping the lables.
    x = df.drop(df.columns[0], axis = 1)
    # dropping all the features
    y = df[df.columns[0]]
    #converts into a numpy matrix.
    y_np = y.values

    #------------TESTS-----------------#
    df_tests = pd.read_csv(test, header=None)
    x_df_test = df_tests.values
    #print("type of test: ",type(x_df_test))
    
    k = df_json["hyperparameters"]["n_neighbors"]
    dist = df_json["hyperparameters"]["distance"]
        
    my_classifier = knn()
    #training the classifier
    my_classifier.fit(x,y_np)
    #...predictions..#
    my_classifier.predict(x,x_df_test,y_np,dist,k)
if __name__ == '__main__':
    main()