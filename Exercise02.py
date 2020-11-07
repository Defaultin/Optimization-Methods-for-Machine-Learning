# looking at data
from sklearn import datasets

wine = datasets.load_wine()
print("Number of entries: ", len(wine.data))
for featurename in wine.feature_names:
    print(featurename[:10], "   \t", end="")

print("Class")
for feature, label in zip(wine.data, wine.target):
    for f in feature:
        print(f, "\t\t",end="")
    print(label)


##################################################
# split data into train and test
from sklearn.model_selection import train_test_split as tts

feats = wine.data
labels = wine.target
train_feats, test_feats, train_labels, test_labels = tts(feats, labels, test_size=0.2)

print("Number of entries: ", len(train_feats))
for featurename in wine.feature_names:
    print(featurename[:10], "   \t",end="")
print("Class")
for feature, label in zip(train_feats,train_labels):
    for f in feature:
        print(f, "\t\t",end="")
    print(label)


##################################################
# svm classifier
from sklearn import svm

clf = svm.SVC(kernel='linear')
clf.fit(train_feats, train_labels)
predictions = clf.predict(test_feats)
print(predictions)
score = sum(predictions == test_labels)
print("SVM: ", score / len(predictions))


###################################################
# tree classifier
from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=3,max_leaf_nodes=3)
clf.fit(train_feats, train_labels)
predictions = clf.predict(test_feats)
print(predictions)
score = sum(predictions == test_labels)
print("Tree: ", score / len(predictions))