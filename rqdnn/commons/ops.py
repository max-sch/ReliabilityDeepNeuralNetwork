import numpy as np

def calc_avg(vec):
    return np.matmul(np.transpose(vec), np.ones((len(vec)))) / len(vec)

def determine_deviation_softmax(softmax, true_labels, class_to_idx_mapper):
    '''
    Calculates the positions of the true label in a softmax output.
    softmax : (N,M) array_like
        Matrix of softmax results.
    y : (N,) array_like
        Input array of the true labels.
    '''
    def create_selection_matrix(pos_encoding, shape):
        # Creates a matrix of ones and zeros where a positional vector encodes the positions of ones of each row.
        m = np.zeros(shape, int)
        m[np.arange(len(pos_encoding)), pos_encoding] = 1
        return m
    
    pos_encoding = np.apply_along_axis(arr=true_labels, 
                                       func1d=lambda x: class_to_idx_mapper(x),
                                       axis=0)
    selection_matrix = create_selection_matrix(pos_encoding, softmax.shape)
    
    softmax_ranks = np.argsort(softmax, axis=1)
    pos_matrix = np.matmul(softmax_ranks, np.transpose(selection_matrix))
    return np.diag(pos_matrix)

def find_type(collection, type):
    for val in collection:
        if isinstance(val, type):
            return val
        
    return None