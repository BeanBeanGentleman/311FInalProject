# TODO: complete this file.
from knn import *
from item_response import *
from neural_network import *

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

    #Generates a random sample 
    resample= np.random.choice(num_entries, num_entries, replace=True)
    
    return {"user_id": id[resample],
            "question_id": q[resample],
            "is_correct": is_correct[resample]}

def evaluate():
    pass
