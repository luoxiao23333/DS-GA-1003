import numpy as np
import scipy.optimize
from scipy.optimize import minimize

root_dir: str = "data//"


def construct_dataset():
    m = 150  # data rows
    d = 75  # feature dimensions
    X: np.ndarray = np.random.rand(m, d)
    theta: np.ndarray = np.zeros(shape=(d, 1))

    theta[:10] = np.array([10 if np.random.randint(0, 2) == 0 else -10 for _ in range(10)]).reshape((10, 1))

    epsilon: np.ndarray = np.random.normal(loc=0, scale=0.1, size=(m, 1))
    y: np.ndarray = np.dot(X, theta) + epsilon

    np.savetxt(root_dir + "X_train.txt", X[:80])
    np.savetxt(root_dir + "X_validation.txt", X[80:100])
    np.savetxt(root_dir + "X_test.txt", X[100:])
    np.savetxt(root_dir + "Y_train.txt", y[:80])
    np.savetxt(root_dir + "Y_validation.txt", y[80:100])
    np.savetxt(root_dir + "Y_test.txt", y[100:])


def ridge_regression_test():
    def ridge(Lambda):
        def ridge_obj(theta):
            return ((np.linalg.norm(np.dot(X_train, theta) - y_train)) ** 2) \
                   / (2 * N) + Lambda * (np.linalg.norm(theta)) ** 2

        return ridge_obj

    def compute_loss(Lambda, theta):
        return ((np.linalg.norm(np.dot(X_valid, theta) - y_valid)) ** 2) / (2 * N)

    X_train = np.loadtxt(root_dir + "X_train.txt")
    X_valid = np.loadtxt(root_dir + "X_validation.txt")
    y_train = np.loadtxt(root_dir + "Y_train.txt")
    y_valid = np.loadtxt(root_dir + "Y_validation.txt")

    (N, D) = X_train.shape
    w = np.random.rand(D)

    min_lambda = 0
    min_loss = 1e100
    min_opt_result: scipy.optimize.OptimizeResult
    for i in range(-5, 6):
        Lambda = 10 ** i
        w_opt = minimize(ridge(Lambda), w)
        loss = compute_loss(Lambda, w_opt.x)
        if loss < min_loss:
            min_loss = loss
            min_opt_result = w_opt
            min_lambda = Lambda
        print(Lambda, loss)
    print("lambda: ", min_lambda, "min loss", min_loss)
    print("theta result is: \n", min_opt_result.x)

    true_zero_count = 0
    thresh_hold = 1e-3
    small_count = 0
    for ele in min_opt_result.x:
        if ele == 0:
            true_zero_count = true_zero_count + 1
        elif ele <= thresh_hold:
            small_count = small_count + 1
    print("True zero count is {0}, values smaller than {1} is {2}"
          .format(true_zero_count, thresh_hold, small_count))
