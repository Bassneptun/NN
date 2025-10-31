from sklearn.neural_network import MLPRegressor
import numpy as np

network = MLPRegressor(solver="sgd", hidden_layer_sizes=(3,3), max_iter=1000, activation="logistic")

network = network.fit(np.array([[0,0],[1,0],[1,1],[0,1]]), np.array([[0],[1],[0],[1]]).flatten())

print(network.score(np.array([[0,0],[1,0],[1,1],[0,1]]), np.array([[0],[1],[0],[1]]).flatten()))
