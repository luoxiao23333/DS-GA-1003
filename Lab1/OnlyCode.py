import configparser
import distutils.log

import numpy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


# Assignment Owner: Tian Wang

#######################################
# Q2.1: Normalization


def feature_normalization(train: np.ndarray, test: np.ndarray) -> [np.ndarray, np.ndarray]:
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """
    min_value: np.ndarray = np.min(train, axis=0)
    max_value: np.ndarray = np.max(train, axis=0)

    for index in range(len(max_value)):
        if min_value[index] == max_value[index]:
            min_value[index] = 0

    train_normalized = (train - min_value) / (max_value - min_value + 0.0)
    test_normalized = (test - min_value) / (max_value - min_value + 0.0)

    return train_normalized, test_normalized


########################################
# Q2.2a: The square loss function

def compute_square_loss(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    Given a set of X, y, theta, compute the square loss for predicting y with X*theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)

    Returns:
        loss - the square loss, scalar
    """

    num_instances = X.shape[0]
    action = np.dot(X, theta)
    difference: np.ndarray = action - y

    loss: float = 0.5 / num_instances * np.dot(difference, difference)
    return loss


########################################
# Q2.2b: compute the gradient of square loss function
def compute_square_loss_gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    """
    num_instances = X.shape[0]
    difference: np.ndarray = np.dot(X, theta) - y
    grad = 1.0 / (num_instances+0.0) * np.dot(difference, X)
    return grad
    """
    num_instance = X.shape[0]
    difference = np.dot(X, theta) - y
    grad = (1.0 / num_instance) * np.dot(difference.T, X)
    return grad


###########################################
# 2.3a: Gradient Checker
# Getting the gradient calculation correct is often the trickiest part
# of any gradient-based optimization algorithm.  Fortunately, it's very
# easy to check that the gradient calculation is correct using the
# definition of gradient.
# See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X: np.ndarray, y: np.ndarray, theta: np.ndarray, epsilon=0.01, tolerance=1e-4) -> bool:
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions:
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1)

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by:
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error

    Return:
        A boolean value indicate whether the gradient is correct or not

    """
    true_gradient: np.ndarray = compute_square_loss_gradient(X, y, theta)  # the true gradient
    num_features = theta.shape[0]
    approx_grad: np.ndarray = np.zeros(num_features)  # Initialize the gradient we approximate
    # TODO
    for index in range(num_features):
        e_i = np.zeros(num_features)
        e_i[index] = 1
        theta_plus = theta + epsilon * e_i
        theta_minus = theta - epsilon * e_i
        approx_grad[index] = \
            (compute_square_loss(X, y, theta_plus) - compute_square_loss(X, y, theta_minus)) \
            / (2 * epsilon)
    distance = np.linalg.norm(approx_grad - true_gradient)
    return distance < tolerance


#################################################
# Q2.3b: Generic Gradient Checker
def generic_gradient_checker(X, y, theta, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. And check whether gradient_func(X, y, theta) returned
    the true gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    true_gradient: np.ndarray = gradient_func(X, y, theta)  # the true gradient
    num_features = theta.shape[0]
    approx_grad: np.ndarray = np.zeros(num_features)  # Initialize the gradient we approximate

    for index in range(num_features):
        e_i = np.zeros(num_features)
        e_i[index] = 1
        theta_plus = theta + epsilon * e_i
        theta_minus = theta - epsilon * e_i
        approx_grad[index] = \
            (objective_func(X, y, theta_plus) - objective_func(X, y, theta_minus)) \
            / (2 * epsilon)
    distance = np.linalg.norm(approx_grad - true_gradient)
    return distance < tolerance


####################################
# Q2.4a: Batch Gradient Descent
def batch_grad_descent(X: np.ndarray, y: np.ndarray, step_size=0.1, num_iter=1000, check_gradient=False) \
        -> [np.ndarray, np.ndarray]:
    """
    In this question you will implement batch gradient descent to
    minimize the square loss objective

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        step - step size in gradient descent
        num_iter - number of iterations to run
        check_gradient - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - store the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features)
                    for instance, theta in iteration 0 should be theta_hist[0], theta in iteration (num_iter) is theta_hist[-1]
        loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    loss_hist = np.zeros(num_iter + 1)  # initialize loss_hist
    theta = np.ones(num_features)  # initialize theta
    # TODO
    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta)

    for iteration in range(1, num_iter + 1):
        if check_gradient:
            assert grad_checker(X, y, theta)
        grad = compute_square_loss_gradient(X, y, theta)
        theta = theta - step_size * grad
        theta_hist[iteration] = theta
        loss = compute_square_loss(X, y, theta)
        loss_hist[iteration] = loss

    return theta_hist, loss_hist


####################################
# Q2.4b: Implement backtracking line search in batch_gradient_descent
# Check http://en.wikipedia.org/wiki/Backtracking_line_search for details
# TODO


###################################################
# Q2.5a: Compute the gradient of Regularized Batch Gradient Descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg) -> np.ndarray:
    """
    Compute the gradient of L2-regularized square loss function given X, y and theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    # TODO
    square_loss_gradient = compute_square_loss_gradient(X, y, theta)
    return square_loss_gradient + (2 * lambda_reg * theta)


###################################################
# Q2.5b: Batch Gradient Descent with regularization term
def regularized_grad_descent(X, y, step_size=0.05, lambda_reg=1, num_iter=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        numIter - number of iterations to run

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features)
        loss_hist - the history of regularized loss value, 1D numpy array
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    loss_hist = np.zeros(num_iter + 1)  # initialize loss_hist
    theta = np.ones(num_features)  # initialize theta
    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta)

    for iteration in range(1, num_iter + 1):
        grad = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
        theta = theta - step_size * grad.T
        theta_hist[iteration] = theta
        loss_hist[iteration] = compute_square_loss(X, y, theta)

    return theta_hist, loss_hist


#############################################
# Q2.5c: Visualization of Regularized Batch Gradient Descent
# X-axis: log(lambda_reg)
# Y-axis: square_loss


def visualize_regularized_batch_gradient_descent(X, y):
    num_iter = 10
    for lambda_reg in [1e-5, 1e-3, 1e-1, 1, 10, 100]:
        theta_hist, loss_hist = regularized_grad_descent(X, y, lambda_reg=lambda_reg, num_iter=num_iter)
        draw(theta_hist, loss_hist, lambda_reg, num_iter)


#############################################
# Q2.6a: Stochastic Gradient Descent
def stochastic_grad_descent(X, y, step_size=0.1, lambda_reg: float = 1, num_iter=100) -> [np.ndarray, np.ndarray]:
    """
    In this question you will implement stochastic gradient descent with a regularization term

    Args:
        :param X - the feature vector, 2D numpy array of size (num_instances, num_features)
        :param y - the label vector, 1D numpy array of size (num_instances)
        :param step_size - string or float. step size in gradient descent
                NOTE: In SGD, it's not always a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if step_size is a float, then the step size in every iteration is alpha.
                if step_size == "1/sqrt(t)", alpha = 1/sqrt(t)
                if step_size == "1/t", alpha = 1/t
                if step_size == "frac", step_size = step_size_0/(1+step_size_0*lambda*t)"
        :param lambda_reg - the regularization coefficient
        :param num_iter - number of epochs (i.e. number of times) to go through the whole training set

    Returns:
        :return theta_hist - the history of parameter vector, 3D numpy array of size (num_iter, num_instances, num_features)
        :return loss hist - the history of regularized loss function vector, 2D numpy array of size(num_iter, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features)  # Initialize theta

    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    loss_hist = np.zeros(num_iter + 1)  # Initialize loss_hist
    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta)
    step_size_0: float = 1.0
    step_size_method = step_size
    if step_size_method == "frac":
        step_size_0: float = 0.1

    for t in range(1, num_iter + 1):
        if step_size_method == "1/sqrt(t)":
            step_size = 1.0 / np.sqrt(t + 1)
        elif step_size_method == "1/t":
            step_size = 1 / (t + 1)
        elif step_size_method == "frac":
            step_size = step_size_0 / (1.0 + step_size_0 * lambda_reg * (t + 1.0))
        else:
            pass

        X_1, X_2, y_1, y_2 = train_test_split(X, y, test_size=0.5, random_state=10)
        X_1, X_2 = feature_normalization(X_1, X_2)
        grad = compute_regularized_square_loss_gradient(X_1, y_1, theta, lambda_reg)
        theta = theta - step_size * grad.T
        grad = compute_regularized_square_loss_gradient(X_2, y_2, theta, lambda_reg)
        theta = theta - step_size * grad.T
        loss_hist[t] = compute_square_loss(X, y, theta) + np.dot(theta, theta) * lambda_reg
        theta_hist[t] = theta

    return theta_hist, loss_hist


def visualize_sgd(X, y):
    num_iter = 1000
    for step_size in [0.05, "1/sqrt(t)", "1/t", "frac"]:
        theta_hist, loss_hist = stochastic_grad_descent(X, y, step_size,
                                                        lambda_reg=1e-5, num_iter=num_iter)
        draw(theta_hist, loss_hist, step_size, num_iter)


################################################
# Q2.6b Visualization that compares the convergence speed of batch
# and stochastic gradient descent for various approaches to step_size
# X-axis: Step number (for gradient descent) or Epoch (for SGD)
# Y-axis: log(objective_function_value)
def visualize_compare_batch_and_sgd(X, y):
    num_iter = 100
    for step_size in [0.001, 0.01, 0.05, 0.1, 0.2]:
        theta_hist_batch, loss_hit_batch = batch_grad_descent(X, y, step_size, num_iter)
        theta_hist_sgd, loss_hit_sgd = stochastic_grad_descent(X, y,
                                                               step_size=step_size,
                                                               num_iter=num_iter,
                                                               lambda_reg=1e-4)
        plt.subplot(1, 2, 1)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss over Step Size = {0}'.format(step_size))
        plt.plot(loss_hit_batch, label="Full Batch")
        plt.plot(loss_hit_sgd, label="SGD")
        if loss_hit_batch[-1] > 1000:
            plt.yscale('log')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.xlabel('Iterations')
        plt.ylabel('Theta')
        plt.title('Theta over Step Size = {0}'.format(step_size))
        plt.plot(theta_hist_batch.transpose()[0], label="Full Batch")
        plt.plot(theta_hist_sgd.transpose()[0], label="SGD")
        if loss_hit_batch[-1] > 1000:
            plt.yscale('log')
        plt.legend()

        plt.show()


def draw(theta_hist, loss_hist, step_size, num_iter):
    x_index = []
    for i in range(num_iter + 1):
        x_index.append(i)
    plt.subplot(1, 2, 1)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss over Step Size')
    plt.plot(x_index, loss_hist, label=step_size)
    if loss_hist[-1] > 1000:
        plt.yscale('log')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x_index, theta_hist.transpose()[0], label=step_size)
    plt.xlabel('Iterations')
    plt.ylabel('Theta')
    plt.title('Theta over Step Size')
    if theta_hist.transpose()[0][-1] > 100:
        plt.yscale('symlog')
    plt.legend()
    plt.show()


def converge_test(X, y, num_iter=100):
    step_sizes = np.array([0.0001, 0.01, 0.05, 0.1, 0.101, 0.2])
    x_index = []
    for i in range(num_iter + 1):
        x_index.append(i)

    for step_size in step_sizes:
        theta_hist, loss_hist = batch_grad_descent(X, y,
                                                   step_size=step_size,
                                                   num_iter=num_iter,
                                                   check_gradient=False)
        draw(theta_hist, loss_hist, step_size, num_iter)


def main():
    # Loading the dataset
    print('loading the dataset')

    df = pd.read_csv('D:\\学术资料\\硕士\\NYU\\DS-GA-1003\\Week1\\hw1\\hw1-sgd\\hw1-data.csv',
                     delimiter=',')
    X = df.values[:, :-1]
    y = df.values[:, -1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)

    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))  # Add bias term
    # TODO
    visualize_compare_batch_and_sgd(X_train, y_train)


if __name__ == "__main__":
    main()
