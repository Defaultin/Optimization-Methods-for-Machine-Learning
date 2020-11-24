from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split as tts
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np


mnist = fetch_openml('mnist_784', version=1)
X_train, X_test, Y_train, Y_test = tts(mnist.data, mnist.target, random_state=0)
X_train = X_train / 255
X_test = X_test / 255


def show_data(data, *, indexes=None, dimension=5, plot_size=(6, 6)):
    '''Render random pictures of numbers'''
    fig = plt.figure(figsize=plot_size)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    plt.title('MNIST dataset')
    plt.axis('off')

    dim = np.ceil(np.sqrt(len(indexes))) if indexes else dimension
    idxs = indexes if indexes else np.random.randint(len(data), size=dim**2)
    for i, pic in enumerate(idxs):
        ax = fig.add_subplot(dim, dim, i+1, xticks=[], yticks=[])
        ax.imshow(mnist.data[pic].reshape(28, 28), cmap=plt.cm.binary, interpolation='nearest')
        ax.text(0, 7, mnist.target[pic])
    plt.show()


def show_weights(clf, *, scale=.5):
    '''Visualize the weight matrices of MLP'''
    fig, axes = plt.subplots(4, 4)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for coef, ax in zip(clf.coefs_[0].T, axes.ravel()):
        ax.matshow(
            coef.reshape(28, 28), 
            cmap=plt.cm.gray, 
            vmin=clf.coefs_[0].min() * scale, 
            vmax=clf.coefs_[0].max() * scale
        )
        ax.set_xticks(())
        ax.set_yticks(())
    plt.show()


def main():
    clf = MLPClassifier(hidden_layer_sizes=2**7, activation='relu', solver='adam', random_state=0)
    clf.fit(X_train, Y_train)
    not_recognized = [i for i, pred in enumerate(clf.predict(X_test)) if pred != Y_test[i]]
    print('Score: ', clf.score(X_test, Y_test))
    print('Total incorrect: ', len(not_recognized))
    show_data(mnist.data, indexes=not_recognized[:16])
    show_weights(clf)


if __name__ == '__main__':
    main()