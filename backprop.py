from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np

network = MLPRegressor(solver="sgd", hidden_layer_sizes=(3,3), max_iter=10000, learning_rate_init=0.3, early_stopping=False, tol=1e-20)

network = network.fit(np.array([[0,0],[1,0],[1,1],[0,1]]), np.array([0,1,0,1]))

print(network.n_iter_)
print(network.loss_curve_)
print(network._no_improvement_count)

plt.title("Neuronales Netz optimiert von Stochastic Gradient Descent")
plt.xlabel("Iterationen")
plt.ylabel("Verlust")
plt.plot(network.loss_curve_, label="loss")
plt.show()
