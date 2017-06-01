from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np


iris=load_iris()

"""
print (iris.feature_names)
print (iris.target_names)
print (iris.data)


"""
#Priting my data
for i in range (len(iris.target)):
	print("{}. {} belongs to {}".format(i+1,iris.data[i],iris.target[i]))

#Test data
test_idx=[0,50,100] #Random indices
test_target=iris.target[test_idx]
test_data=iris.data[[0,50,100]]

#Training Data
train_target=np.delete(iris.target,test_idx)
train_data=np.delete(iris.data,test_idx,axis=0)



clf=tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)

#Predicting what we trained on testing data

print (clf.predict(test_data))
#Just to verify if our prediction is right!
print (test_target) 