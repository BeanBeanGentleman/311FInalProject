from sklearn.impute import KNNImputer
from utils import *
import os
import numpy as np
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    acc = None
    neighbours = KNNImputer(n_neighbors=k)
    t_mat = np.transpose(neighbours.fit_transform(np.transpose(matrix)))
    acc = sparse_matrix_evaluate(valid_data, t_mat)
    print("Validation Accuracy: {}".format(acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_values = [1, 6, 11, 16, 21, 26]
    acc_u = []
    acc_i = []
    for k in k_values:
        uaccuracy = knn_impute_by_user(sparse_matrix, val_data, k)
        iaccuracy = knn_impute_by_item(sparse_matrix, val_data, k)
        acc_u.append(uaccuracy)
        acc_i.append(iaccuracy)
    
    best_u = np.argmax(acc_u)
    best_i = np.argmax(acc_i)

    ku = k_values[best_u]
    ki = k_values[best_i]

    print("When k is {}, test has the highest accuracy {}.".format(ku, acc_u[best_u]))
    print("When k is {}, test has the highest accuracy {}.".format(ki, acc_i[best_i]))
    
    # plt.plot(k_values, acc_i, color='blue')
    # plt.xlabel("K Value")
    # plt.ylabel("Accuracy")
    # plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
