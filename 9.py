import numpy as np
import matplotlib.pyplot as plt

def locally_weighted_regression(X, y, tau):
    m = X.shape[0]
    y_pred = np.zeros(m)
    
    for i in range(m):
        w = np.exp(-np.sum((X - X[i, :])**2, axis=1) / (2 * tau**2))
        W = np.diag(w)
        theta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
        y_pred[i] = X[i, :] @ theta
    
    return y_pred

# Example usage:
X = np.array([[1, x] for x in np.linspace(1, 10, 10)])
y = np.array([1.0, 2.2, 2.8, 4.4, 5.0, 6.1, 7.2, 7.8, 9.1, 10.2])
tau = 0.5
y_pred = locally_weighted_regression(X, y, tau)

plt.scatter(X[:, 1], y, color='blue', label='Data')
plt.plot(X[:, 1], y_pred, color='red', label='Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
