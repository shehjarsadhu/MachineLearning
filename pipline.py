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

from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

#training the classifier
my_classifier.fit(X_train,Y_train)

predictions = my_classifier.predict(X_test)

print(predictions)

#How accurate the test is ?
#Comapre the predicted lables to the ture lables and tally the score.
from sklearn.metrics import accuracy_score

print(accuracy_score(Y_test, predictions))
