# TODO: complete this file.
from utils import *
from item_response import *
import torch

def sample(train):
    '''
    return the bootstrap sample with a sample size same as size of the input training set
    '''

    id = np.array(train['user_id'])
    q = np.array(train['question_id'])
    is_correct = np.array(train['is_correct'])

    assert len(id) == len(q)
    assert len(q) == len(is_correct)

    num_entries = len(id)

    #draw a sample with replacement with the chosen size
    resample= np.random.choice(num_entries, num_entries, replace=True)
    
    return {"user_id": id[resample],
            "question_id": q[resample],
            "is_correct": is_correct[resample]}

def ensemble_evaluate(data, theta, beta):
    '''
    For each bootstrap sample, estimate the skill of the model, calculate the mean of the sample of model skill estimates,
    and then return the accuracy.
    '''
    pred1 = []
    for i, q in enumerate(data["question_id"]):
        pred2 = []
        id = data["user_id"][i]
        for k in range(len(theta)):
            pred2.append(sigmoid((theta[k][id] - beta[k][q]).sum()))
        pred1.append(np.mean(pred2) >= 0.5)
    denominator = len(data["is_correct"])
    numerator = np.sum((data["is_correct"] == np.array(pred1)))
    return  numerator / denominator
        

if __name__ == '__main__':
    train = load_train_csv("../data")
    val = load_valid_csv("../data")
    test = load_public_test_csv("../data")

    # Select and train 3 base models with bootstrapping the training set.
    print("-----Training the first model with bootstrap-----")
    theta_1, beta_1, val_acc_lst_1, train_likelihood_1, val_likelihood_1 \
        = irt(sample(train), val, 0.01, 50)

    print()

    print("-----Training the second model with bootstrap-----")
    theta_2, beta_2, val_acc_lst_2, train_likelihood_2, val_likelihood_2 \
        = irt(sample(train), val, 0.01, 50)

    print()

    print("-----Training the third model with bootstrap-----")
    theta_3, beta_3, val_acc_lst_3, train_likelihood_3, val_likelihood_3 \
        = irt(sample(train), val, 0.01, 50)

    print()

    theta = [theta_1, theta_2, theta_3]
    beta = [beta_1, beta_2, beta_3]

    # To predict the correctness, generate 3 predictions by using the base model and average the predicted correctness.

    print("-----Evaluating each individual base model-----")
    
    print()
    print()

    print("Validation accuracy for each individual base model are:", 
    evaluate(val, theta_1, beta_1),
    evaluate(val, theta_2, beta_2), 'and',
    evaluate(val, theta_3, beta_3)) 
    
    print("Test accuracy for each individual base model are:", 
    evaluate(test, theta_1, beta_1),
    evaluate(test, theta_2, beta_2), 'and',
    evaluate(test, theta_3, beta_3)) 

    print()
    
    print("-----Report final accuracy-----")

    print()
    print()
    final_val_accuracy = ensemble_evaluate(val, theta, beta)
    final_test_accuracy = ensemble_evaluate(test, theta, beta)
    print("Final validation accuracy is: ", final_val_accuracy)
    print("Final test accuracy is: ", final_test_accuracy)