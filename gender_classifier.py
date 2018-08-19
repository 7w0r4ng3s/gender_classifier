from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

clf1 = tree.DecisionTreeClassifier()
clf2 = RandomForestClassifier()


# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], 
     [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# train them on our data
clf1 = clf1.fit(X, Y)
clf2.fit(X, Y)

prediction = clf1.predict([[190, 70, 43]])
prediction2 = clf2.predict([[140, 50, 43]])

# compare their reusults and print the best one!

print(prediction)
print(prediction2)
