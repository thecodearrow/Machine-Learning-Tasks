from sklearn import tree

features=[[100,1],[70,1],[50,0],[52,0]]
labels=[1,1,0,0]
clf=tree.DecisionTreeClassifier()
clf=clf.fit(features,labels)
print (clf.predict([[90,1]]))
