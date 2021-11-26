from utils import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.

    users = data['user_id']
    questions = data['question_id']
    correct = data['is_correct']

    for i, c in enumerate(correct):
        user = users[i]
        question = questions[i]

        log_lklihood += c * (theta[user] - beta[question]) - np.log(1 + np.exp(theta[user] - beta[question]))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    users = data['user_id']
    questions = data['question_id']
    correct = data['is_correct']

    result_beta = [0 for z in range(len(beta))]
    result_theta = [0 for z in range(len(theta))]
    for i, c in enumerate(correct):
        user = users[i]
        question = questions[i]
        value = c - sigmoid(theta[user] - beta[question])
        result_beta[question] += value
        result_theta[user] += value
    
    theta += lr * np.array(result_theta)
    beta -= lr * np.array(result_beta)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.random.rand(542)
    beta = np.random.rand(1774)

    val_acc_lst = []
    train_likelihood = []
    val_likelihood = []


    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        # print("NLLK: {} \t Score: {}".format(neg_lld, score))

        neg_lld_val = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_likelihood.append(neg_lld_val)
        train_likelihood.append(neg_lld)

        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_likelihood, val_likelihood


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################

    # learning_rate = [0.1, 0.05, 0.01, 0.005, 0.001]
    # iterations = [10, 20, 30, 50, 100]
    # for lr in learning_rate:
    #     for iter in iterations:
    #         theta, beat, val_acc_lst, train_likelihood, val_likelihood = irt(train_data, val_data, lr, iter)
    #         print("Current lr: {}, iterations: {}, accuracy: {}".format(lr, iter, val_acc_lst[-1]))
    # Best 0.01, 20
    theta, beta, val_acc_lst, train_likelihood, val_likelihood = irt(train_data, val_data, 0.01, 20)
    # plt.plot(train_likelihood, label='train')
    # plt.plot(val_likelihood, label='val')
    # plt.xlabel('iterations number')
    # plt.ylabel('Loglikelihood')
    # x_major_locator = MultipleLocator(1)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.legend()
    # plt.show()
    test_acc = evaluate(test_data, theta, beta)
    print("Test accuracy: ", test_acc)
    val_acc = evaluate(val_data, theta, beta)
    print("Validation accuracy: ", val_acc)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    value_1 = sigmoid(theta - beta[1])
    value_2 = sigmoid(theta - beta[5])
    value_3 = sigmoid(theta - beta[20])

    plt.scatter(theta, value_1, label='j1')
    plt.scatter(theta, value_2, label='j2')
    plt.scatter(theta, value_3, label='j3')

    plt.xlabel("Theta")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
