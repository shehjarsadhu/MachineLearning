import random
#inctalled by anaconda
from scipy.spatial import distance
#Retuens the euclidean distance.
def euc(a,b):
        return distance.euclidean(a,b)
class scrapyKNN():
    #takes features and lables of the traning set.
    def fit(self,X_train,Y_train):
        #storing the traning data in the class
        #Like Memorising the data.
        #these are the features...
        self.X_train = X_train
        #These are the lables
        self.Y_train = Y_train
    #    print("This is-----------------self.X_train----------------" )
    #    print(self.X_train)
    #    print("This is-----------------self.Y_train----------------" )
    #    print(self.Y_train)

        #recives features of the testing data.
        #Returns prediction for the lables
    def predict(self,X_test):
        #Returns list of predictions.
        #X_test => 2d Array
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    def closest(self,row):
        #calc. dist b/w test pt to the 1st traning pt.
        best_dist = euc(row, self.X_train[0])
        #keep track of the index of the traning point that is closest.
        best_index = 0
        #loop over each traning point(features)
        for i in range(1,len(self.X_train)):
                dist = euc(row,self.X_train[i])
                #finding the closest distance.
                if dist < best_dist:
                     best_dist = dist
                     best_index = i
        #return the lable to the closest traning example.
        return self.Y_train[best_index]


from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y= iris.target

#partition the data into train and testing

from sklearn.cross_validation import train_test_split

#X_train ,Y_train => (features , lables) for training
#X_test, Y_test => features , lables) for testing.
#75% of the data is in train
X_train, X_test, Y_train, Y_test =train_test_split(X, y, test_size = 0.5)

#Creating a classifier
#By changing the classifier the accuracy score can be changed.

#from sklearn import tree
#my_classifier = tree.DecisionTreeClassifier()

#from sklearn.neighbors import KNeighborsClassifier
my_classifier = scrapyKNN()

#training the classifier
my_classifier.fit(X_train,Y_train)

predictions = my_classifier.predict(X_test)

print(predictions)

#How accurate the test is ?
#Comapre the predicted lables to the ture lables and tally the score.
from sklearn.metrics import accuracy_score

print(accuracy_score(Y_test, predictions))
