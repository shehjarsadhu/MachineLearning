# -----------Decision tree--------Iris-------------#

# 3. Predict label for new flower...

# 1. Import data set
from sklearn.datasets import load_iris
iris = load_iris()
#Features of the flower
print(iris.feature_names)
#More like lables names of flower
print(iris.target_names)
# data at inex 0. feartres of flower one.
print(iris.data[0])

#for i in range(len(iris.target)):
#    print("Example %d: lable %s:, features %s lab" % i, iris.target[i],iris.target[i],iris.data[i])


# 2. Train the classifier:
import numpy as np

#removing examples.
#first setosa , versicolor, virginica...
test_idx = [0,50,100]
#Traning data
#has majority of the data
train_target = np.delete(iris.target, test_idx)
train_data  = np.delete(iris.data, test_idx, axis=0)


#Testing data
#Includes onl the examples removed.
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)

print(test_target)
print(clf.predict(test_data))
#testing ecample
print(test_data[0],test_target[0])
print(test_data[1],test_target[1])
