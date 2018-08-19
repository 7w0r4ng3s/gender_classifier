from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# data for training
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], 
     [181, 85, 43]]

y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# data for testing
_X=[[184,84,44], [198,92,48], [183,83,44], [166,47,36], [170,60,38], [172,64,39],
	[182,80,42],[180,80,43]]
_y=['male', 'male', 'male', 'female', 'female', 'female', 'male', 'male']


# classifiers
clf_tree = tree.DecisionTreeClassifier()
clf_knn = KNeighborsClassifier()
clf_svm = SVC()
clf_mlp = MLPClassifier()

# train models using data above
clf_tree.fit(X, y)
clf_knn.fit(X, y)
clf_svm.fit(X, y)
clf_mlp.fit(X, y)

# prediction
pred_tree = clf_tree.predict(_X)
pred_knn = clf_knn.predict(_X)
pred_svm = clf_svm.predict(_X)
pred_mlp = clf_svm.predict(_X)

# accuracy scores
score_tree = accuracy_score(_y, pred_tree) * 100
score_knn = accuracy_score(_y, pred_knn) * 100
score_svm = accuracy_score(_y, pred_svm) * 100
score_mlp = accuracy_score(_y, pred_mlp) * 100

print("tree: ", score_tree)
print("knn: ", score_knn)
print("svm: ", score_svm)
print("mlp: ", score_mlp)