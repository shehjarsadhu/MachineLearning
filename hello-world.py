from sklearn import tree
#input to the classifier.
#1==>Smooth ,0=>Bumpy
features = [[140,1],[130,1],[150,0],[170,0]]
# 0=> Apple, 1=>Orange
lables = [0,0,1,1]
#Clasifier is like a box of rules....
#Train a classifier. Here classifier ==> Decision Tree.
#Creating the classifier.We used Decision tree in this case
clf = tree.DecisionTreeClassifier()
#Learning algo is the procedure that creates them....
#the traning algo that gets fed to the classifier as
clf = clf.fit(features, lables)
#Input to the classifier is the fratures of the input.
print(clf.predict([[150 , 0]]))
