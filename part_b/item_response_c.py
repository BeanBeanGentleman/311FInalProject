from utils import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import warnings
warnings.filterwarnings("error")
from datetime import datetime
import pandas as pd

g = 0.03

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
        miu = np.exp(miu)
        try:
            log_lklihood += c * np.log((g + miu) / (1 - g)) + np.log((1 - g) / (1 + miu))
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
        miu = alpha[question] * (theta[user] - beta[question])
        miu = np.exp(miu)
        value = c / (miu + g) - 1 / (miu + 1)
        result_beta[question] += alpha[question] * value * miu
        result_theta[user] += alpha[question] * value * miu
        result_alpha[question] += (theta[user] - beta[question]) * value * miu

    
    theta += lr * np.array(result_theta)
    beta -= lr * np.array(result_beta)
    alpha += lr * np.array(result_alpha)
    # import pdb; pdb.set_trace()   

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta, alpha


def irt(data, val_data, lr, iterations, theta):
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
    # theta = np.zeros(542)
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
        # p_a = sigmoid(x)
        p_a = g + (1 - g) * sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])

def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    meta_data = pd.read_csv("../data/student_meta.csv")

    ftheta, fbeta, falpha, fval_acc_lst, ftrain_likelihood, fval_likelihood = irt(train_data, val_data, 0.005, 30, np.zeros(542))

    tdf = pd.DataFrame(ftheta, columns=['theta'])
    tdf['user_id'] = tdf.index

    now = pd.Timestamp('now')
    meta_data["data_of_birth"] = pd.to_datetime(meta_data["data_of_birth"])
    df = meta_data.loc[:, ['user_id', 'data_of_birth']]
    df = df[df["data_of_birth"] < datetime.now()]
    df["data_of_birth"] = df["data_of_birth"].where(df["data_of_birth"] < now, df["data_of_birth"] -  np.timedelta64(100, 'Y'))   # 2
    df['age'] = (now - df["data_of_birth"]).astype('<m8[Y]')
    # df['theta_weight'] = 0.01 * (df['age'] - df['age'].mean() + 2)
    tdf = pd.merge(tdf, df, how='left', on='user_id')
    w_dict = tdf.groupby('age')['theta'].mean().reset_index().set_index('age').to_dict()['theta']
    # import pdb; pdb.set_trace()
    
    index_lst = (meta_data[meta_data["premium_pupil"] == 1]["user_id"]).to_list()
    input_theta = [0] * 542
    for idx in index_lst:
        input_theta[idx] -= 0.05
    for index, row in df.iterrows():
        uid = row['user_id']
        age = row['age']
        input_theta[uid] += w_dict[age] * 0.005
    input_theta = np.array(input_theta)

    # import pdb; pdb.set_trace()
    

    # learning_rate = [0.003]
    # iterations = [50]
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
    
    # Best 0.003, 40
    theta, beta, alpha, val_acc_lst, train_likelihood, val_likelihood = irt(train_data, val_data, 0.005, 30, input_theta)

    test_acc = evaluate(test_data, theta, beta, alpha)
    print("Test accuracy: ", test_acc)
    val_acc = evaluate(val_data, theta, beta, alpha)
    print("Validation accuracy: ", val_acc)


if __name__ == "__main__":
    main()
