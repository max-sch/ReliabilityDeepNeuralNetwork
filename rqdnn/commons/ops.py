import numpy as np

def calc_avg(vec):
    n = len(vec)
    if n == 0:
        return 0
    return np.matmul(np.transpose(vec), np.ones((n))) / n

def determine_deviation_softmax(softmax, true_labels, class_to_idx_mapper):
    '''
    Calculates the positions of the true label in a softmax output.
    softmax : (N,M) array_like
        Matrix of softmax results.
    y : (N,) array_like
        Input array of the true labels.
    '''
    idxs = np.apply_along_axis(arr=true_labels, 
                               func1d=lambda x: class_to_idx_mapper(x),
                               axis=0)
    idxs = idxs.reshape((idxs.shape[0], 1))

    softmax_ranks = np.argsort(softmax, axis=1)[:, ::-1]
    positions = softmax_ranks == idxs
    idx_matrix = np.array([np.arange(positions.shape[1])] * positions.shape[0])
    return idx_matrix[positions]

def find_type(collection, type):
    for val in collection:
        if isinstance(val, type):
            return val
        
    return None

def random_splits(splits):
    idxs = []
    for i, split in enumerate(splits):
        idxs = idxs + [i] * split
    
    idxs = np.array(idxs)
    np.random.shuffle(idxs)

    return idxs