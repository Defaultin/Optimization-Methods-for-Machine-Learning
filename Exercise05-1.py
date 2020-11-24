import numpy as np
import time
from itertools import islice
from scipy import sparse
from sklearn import datasets

# Defining some kernels to check psd property
def tanh_kernel(X, Y, c=0):
    return np.tanh(np.dot(X, Y.T) + c)

def linear_kernel(X, Y):
    return np.dot(X, Y.T)

def rbf_kernel(X, Y):
    return np.exp(-np.dot(X - Y, (X - Y).T))

# Compute Gram matrix K=(kernel(x_i,x_j))_(i,j)
def computeGram(feats, kernel):
    n = feats.shape[0]
    gram = np.zeros([n, n])
    for i, x in enumerate(feats):
        for j, y in islice(enumerate(feats), i, None):
            gram[i, j] = kernel(x.todense(), y.todense())
            
    low_tri = np.tril_indices(n) # only compute triangular and mirror it to save computing time
    gram[low_tri] = gram.T[low_tri]

    return gram

# Compute smallest eigenvalue
def min_ev_gramian(data,kernel):
    time_gram = time.time()
    gram = computeGram(data, kernel) # build Gram matrix
    print(f'Building Gram Matrix needs: {time.time() - time_gram} seconds')
    
    time_ev = time.time()
    min_eval = np.linalg.eigvals(gram).min()
    print(f'Numpy found min_eval of: {min_eval} in {time.time() - time_ev} seconds')
    
    # alternative approach to check positive definite is to check if Cholesky decomposition works (add small identy matrix to check psd)
    try:
        # will raise an error if Cholesky decomposition can not be computed -> not psd
        cholesky = np.linalg.cholesky(gram + 0.00000001 * np.eye(gram.shape[0]))
    except Exception:
        print('Not psd!')

# Compute Gram matrices and their smalles eigenvalue for some of the datasets in sklearn
wine = datasets.load_wine()
wine_feats = wine.data
iris = datasets.load_iris()
iris_feats = iris.data
print('Tanh kernel:')
min_ev_gramian(sparse.csc_matrix(wine_feats.T), tanh_kernel)
min_ev_gramian(sparse.csc_matrix(iris_feats.T), tanh_kernel)

print('\nLinear kernel:')
min_ev_gramian(sparse.csc_matrix(wine_feats.T), linear_kernel)
min_ev_gramian(sparse.csc_matrix(iris_feats.T), linear_kernel)

print('\nRbf kernel:')
min_ev_gramian(sparse.csc_matrix(wine_feats.T), rbf_kernel)
min_ev_gramian(sparse.csc_matrix(iris_feats.T), rbf_kernel)