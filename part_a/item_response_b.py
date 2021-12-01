from utils import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import warnings
warnings.filterwarnings("error")


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, alpha):
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
        miu = alpha[question] * (theta[user] - beta[question])
        try:
            log_lklihood += c * miu - np.log(1 + np.exp(miu))
        except RuntimeWarning:
            import pdb; pdb.set_trace()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, alpha):
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
    result_alpha = [0 for z in range(len(alpha))]
    for i, c in enumerate(correct):
        user = users[i]
        question = questions[i]
        value = c - sigmoid(alpha[question] * (theta[user] - beta[question]))
        result_beta[question] += alpha[question] * value
        result_theta[user] += alpha[question] * value
        result_alpha[question] += (theta[user] - beta[question]) * value

    
    theta += lr * np.array(result_theta)
    beta -= lr * np.array(result_beta)
    alpha += lr * np.array(result_alpha)
    # import pdb; pdb.set_trace()   

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta, alpha


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
    theta = np.zeros(542)
    beta = np.zeros(1774)
    alpha = np.ones(1774)

    val_acc_lst = []
    train_likelihood = []
    val_likelihood = []


    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, alpha=alpha)
        score = evaluate(data=val_data, theta=theta, beta=beta, alpha=alpha)
        val_acc_lst.append(score)
        # print("NLLK: {} \t Score: {}".format(neg_lld, score))

        neg_lld_val = neg_log_likelihood(val_data, theta=theta, beta=beta, alpha=alpha)
        val_likelihood.append(neg_lld_val)
        train_likelihood.append(neg_lld)

        theta, beta, alpha = update_theta_beta(data, lr, theta, beta, alpha)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, alpha, val_acc_lst, train_likelihood, val_likelihood


def evaluate(data, theta, beta, alpha):
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
        x = (alpha[q] * (theta[u] - beta[q])).sum()
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

    # learning_rate = [0.005, 0.003, 0.001]
    # iterations = [10, 20, 30, 50, 60, 75]
    # best_acc = 0
    # best_iter = 10
    # best_lr = 0.1
    # for lr in learning_rate:
    #     for iter in iterations:
    #         theta, beta, alpha, val_acc_lst, train_likelihood, val_likelihood = irt(train_data, val_data, lr, iter)
    #         if val_acc_lst[-1] > best_acc:
    #             best_acc = val_acc_lst[-1]
    #             best_iter = iter
    #             best_lr = lr
    #         print("Current lr: {}, iterations: {}, accuracy: {}".format(lr, iter, val_acc_lst[-1]))
    # print("Best lr: {}, iterations: {}, accuracy: {}".format(best_lr, best_iter, best_acc))
    
    # Best 0.005, 30
    theta, beta, alpha, val_acc_lst, train_likelihood, val_likelihood = irt(train_data, val_data, 0.005, 30)
    plt.plot(train_likelihood, label='train')
    plt.plot(val_likelihood, label='val')
    plt.xlabel('iterations number')
    plt.ylabel('Loglikelihood')
    x_major_locator = MultipleLocator(2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.legend()
    plt.show()
    test_acc = evaluate(test_data, theta, beta, alpha)
    print("Test accuracy: ", test_acc)
    val_acc = evaluate(val_data, theta, beta, alpha)
    print("Validation accuracy: ", val_acc)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    # value_1 = sigmoid(theta - beta[1])
    # value_2 = sigmoid(theta - beta[5])
    # value_3 = sigmoid(theta - beta[20])

    # plt.scatter(theta, value_1, label='j1')
    # plt.scatter(theta, value_2, label='j2')
    # plt.scatter(theta, value_3, label='j3')

    # plt.xlabel("Theta")
    # plt.ylabel("Probability")
    # plt.legend()
    # plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
