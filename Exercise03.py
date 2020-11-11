import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as tts


iris = load_iris()
feats = iris.data
labels = iris.target
feats_sepal = feats[:, :2]
feats_petal = feats[:, 2:]


def svm_model(feats, labels, *, kernel='rbf', gamma='auto', fine=1, degree=3, random_state=None): 
    split_data = tts(feats, labels, random_state=random_state)
    model = SVC(kernel=kernel, gamma=gamma, C=fine, degree=degree)
    fitted_model = model.fit(split_data[0], split_data[2])
    return split_data, fitted_model


def meshgrid(x, y, h):
    x_lim = x.min() - 1, x.max() + 1
    y_lim = y.min() - 1, y.max() + 1
    x_space = np.arange(*x_lim, h)
    y_space = np.arange(*y_lim, h)
    return np.meshgrid(x_space, y_space)


def svm_2d_contour_plot(split_data, model, *, mesh_granularity=.01):
    x = np.concatenate((split_data[0][:, 0], split_data[1][:, 0]))
    y = np.concatenate((split_data[0][:, 1], split_data[1][:, 1]))
    l = np.concatenate((split_data[2], split_data[3]))
    acc = accuracy_score(l, model.predict(np.c_[x, y]))
    
    X, Y = meshgrid(x, y, mesh_granularity)
    Z = model.predict(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)

    plt.figure()
    plt.contourf(X, Y, Z, alpha=0.4, cmap='rainbow')
    pnts = plt.scatter(x, y, c=l, s=50, cmap='rainbow')
    test_pnts = plt.scatter(split_data[1][:, 0], split_data[1][:, 1], s=100, facecolors='none', edgecolors='black')
    plt.legend([*pnts.legend_elements()[0], test_pnts], ['Setosa', 'Versicolour', 'Virginica', 'Test points'])
    plt.title(f'Kernel: {model.kernel}. C: {model.C}. Gamma: {model.gamma}. Accuracy: {round(acc, 2)}.')


def main():
    svm_2d_contour_plot(*svm_model(feats_sepal, labels, kernel='rbf', random_state=0))
    svm_2d_contour_plot(*svm_model(feats_petal, labels, kernel='poly', degree=5, random_state=0))


if __name__ == '__main__':
    main()