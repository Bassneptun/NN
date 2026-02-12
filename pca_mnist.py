from sklearn.datasets import fetch_openml, load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split

import numpy as np

digits: Bunch = load_digits()
X, Y = fetch_openml("mnist_784", version=1, return_X_y=True)

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

"""
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = Y[:60000], Y[60000:]
"""

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

p = 8
pca = PCA(n_components=p)
X_train_pca: np.ndarray = pca.fit_transform(X_train)
X_test_pca: np.ndarray = pca.transform(X_test)

np.savetxt("X_train.csv", X_train_pca, delimiter=",")
np.savetxt("X_test.csv", X_test_pca, delimiter=",")
np.savetxt("Y_train.csv", y_train, delimiter=",")
np.savetxt("Y_test.csv", y_test, delimiter=",")
