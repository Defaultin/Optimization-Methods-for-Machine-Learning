# import data
from sklearn import datasets
wine = datasets.load_wine()

# looking at data
print("Number of entries: ", len(wine.data))
for featurename in wine.feature_names:
    print(featurename[:10], "   \t",end="")
print("Class")
for feature, label in zip(wine.data,wine.target):
    for f in feature:
        print(f, "\t\t",end="")
    print(label)

# setting up data
feats = wine.data
labels = wine.target

# getting ready for ml
from sklearn.model_selection import train_test_split as tts

# RANDOMLY(!) split data into train and test
train_feats, test_feats, train_labels, test_labels = tts(feats, labels, test_size=0.2)

# looking at the train set
print("Number of entries: ", len(train_feats))
for featurename in wine.feature_names:
    print(featurename[:10], "   \t",end="")
print("Class")
for feature, label in zip(train_feats,train_labels):
    for f in feature:
        print(f, "\t\t",end="")
    print(label)

##################################################

# import svm
from sklearn import svm

# choose classifier
clf = svm.SVC(kernel='linear')

# train the classifier on the train set
clf.fit(train_feats, train_labels)

# predict the classes of the test set
predictions = clf.predict(test_feats)

# print predictions and ratio of succes
print(predictions)

#score = 0
#for i in range(len(predictions)):
#    if predictions[i] == test_labels[i]:
#        score += 1
score = sum(predictions == test_labels)
print("SVM: ", score / len(predictions))

###################################################

# import tree
from sklearn import tree

# choose classifier
clf = tree.DecisionTreeClassifier(max_depth=3,max_leaf_nodes=3)

# train the classifier
clf.fit(train_feats, train_labels)

# predict the classes
predictions = clf.predict(test_feats)

print(predictions)

score = sum(predictions == test_labels)
print("Tree: ", score / len(predictions))