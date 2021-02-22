import time
import numpy as np
from scipy.special import softmax
import torch
from copy import deepcopy

DEFAULT_MEMO = dict()

def computeGramMatrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.
    
    Parameters:
      A:  a (n_batch, n, d) Tensor.
      B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """
    
    assert(A.dim() == 3)
    assert(B.dim() == 3)
    assert(A.size(0) == B.size(0) and A.size(2) == B.size(2))

    return torch.bmm(A, B.transpose(1,2))


def binv(b_mat):
    """
    Computes an inverse of each matrix in the batch.
    Pytorch 0.4.1 does not support batched matrix inverse.
    Hence, we are solving AX=I.
    
    Parameters:
      b_mat:  a (n_batch, n, n) Tensor.
    Returns: a (n_batch, n, n) Tensor.
    """

    id_matrix = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat).cuda()
    b_inv, _ = torch.gesv(id_matrix, b_mat)
    
    return b_inv


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies


def batched_kronecker(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
    matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
    return torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1)).reshape([matrix1.size()[0]] + list(matrix1.size()[1:]) + list(matrix2.size()[1:])).permute([0, 1, 3, 2, 4]).reshape(matrix1.size(0), matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))



def copy_and_replace(original, replace=None, do_not_copy=None):
    """
    A convenience function for creating modified copies out-of-place with deepcopy
    :param original: object to be copied
    :param replace: a dictionary {old object -> new object}, replace all occurences of old object with new object
    :param do_not_copy: a sequence of objects that will not be copied (but may be replaced)
    :return: a copy of obj with replacements
    """
    replace, do_not_copy = replace or {}, do_not_copy or {}
    memo = dict(DEFAULT_MEMO)
    for item in do_not_copy:
        memo[id(item)] = item
    for item, replacement in replace.items():
        memo[id(item)] = replacement
    return deepcopy(original, memo)



def logistic_regression_grad_with_respect_to_w(X, y, w):
    '''
    X, y, w numpy arrays
       shape
    X  N, (d+1) last dimension the bias
    y  N integer class identity
    w  C, (d+1) number of classes C

    return grad C*(d+1), 1
    '''
    preds = np.matmul(X, w.T)
    p = softmax(preds, axis=1) # the probability matrix N, C

    C = w.shape[0]
    N = X.shape[0]

    result = np.zeros((C * X.shape[1], 1), dtype=np.float32)
    I = np.eye(C, dtype=np.float32)
    for i in range(X.shape[0]):
        result += np.kron(p[i].reshape(-1, 1) - I[:, y[i]: y[i]+1], X[i].reshape(-1, 1))
    result /= N
    
    return result


def logistic_regression_hessian_pieces_with_respect_to_w(X, y, w):
    '''
    X, y, w numpy arrays
       shape
    X  N, (d+1) last dimension the bias
    y  N integer class identity
    w  C, (d+1) number of classes C
    
    return hessian components:
    diag N(C+1) diag has been divided by N for each component
    X    N(C+1), C(d+1)
    the final hessian can be computed as X.T @ np.diag(diag) @ X
    '''
    preds = np.matmul(X, w.T)
    p = softmax(preds, axis=1) # the probability matrix N, C

    C = w.shape[0]
    N = X.shape[0]
    d = X.shape[1] - 1
    
    Xbar = np.zeros(shape=(N * (C+1), C * (d+1)), dtype=np.float32)
    diag = []
    
    for i in range(N):
        for j in range(C):
            Xbar[i * C + j, j * (d+1): (j+1) * (d+1)] = X[i]        
        diag.extend(p[i] / N)
    for i in range(N):
        Xbar[N * C + i, :] = np.kron(p[i], X[i])
        diag.append(-1 / N)

    return diag, Xbar


def logistic_regression_hessian_with_respect_to_w(X, y, w):
    '''
    X, y, w numpy arrays
       shape
    X  N, (d+1) last dimension the bias
    y  N integer class identity
    w  C, (d+1) number of classes C
    
    return hessian matrix C*(d+1), C*(d+1)
    '''
    diag, Xbar = logistic_regression_hessian_pieces_with_respect_to_w(X, y, w)
    result = np.matmul(np.matmul(Xbar.T, np.diag(diag)), Xbar)

    return result

def logistic_regression_hessian_old_with_respect_to_w(X, y, w):
    # currently not used
    '''
    X, y, w numpy arrays
       shape
    X  N, (d+1) last dimension the bias
    y  N integer class identity
    w  C, (d+1) number of classes C
    
    return hessian matrix C*(d+1), C*(d+1)
    '''
    preds = np.matmul(X, w.T)
    p = softmax(preds, axis=1) # the probability matrix N, C
    
    C = w.shape[0]
    N = X.shape[0]
    
    result = np.zeros((C * X.shape[1], C * X.shape[1]), dtype=np.float32)
    for i in range(N):
        result += np.kron(np.diag(p[i]) - np.outer(p[i], p[i]), np.outer(X[i], X[i]))
    result /= N

    return result


def logistic_regression_mixed_derivatives_with_respect_to_w_then_to_X(X, y, w):
    '''
    X, y, w numpy arrays
       shape
    X  N, (d+1) last dimension the bias
    y  N integer class identity
    w  C, (d+1) number of classes C

    return mixed partial matrix C*(d+1), N*(d+1)
    '''
    preds = np.matmul(X, w.T)
    p = softmax(preds, axis=1) # the probability matrix N, C
    
    C = w.shape[0]
    N = X.shape[0]
    
    d = X.shape[1] - 1
    
    I_C = np.eye(C, dtype=np.float32)
    I_dp1 = np.eye(d + 1, dtype=np.float32)

    result = np.kron((p - I_C[y]).T, I_dp1)
    
    for i in range(N):
        weighted_w = np.matmul(np.diag(p[i]), np.matmul(I_C - p[i], w)) # broadcasting
        for j in range(C):
            result[j*(d+1):(j+1)*(d+1), i*(d+1): (i+1)*(d+1)] += np.outer(X[i], weighted_w[j])
    result /= N

    return result

def logistic_regression_mixed_derivatives_with_respect_to_w_then_to_X_left_multiply(X, y, w, a):
    '''
    X, y, w, v numpy arrays
       shape
    X  N, (d+1) last dimension the bias
    y  N integer class identity
    w  C, (d+1) number of classes C
    a  C(d+1), 1 (column vector)

    return a^T @ mixed partial matrix of shape N(d+1)
    '''
    preds = np.matmul(X, w.T)
    p = softmax(preds, axis=1) # the probability matrix N, C
    
    C = w.shape[0]
    N = X.shape[0]
    d = X.shape[1] - 1

    a_reshape = a.reshape(C, d + 1)
    weighted_a = np.matmul(p, a_reshape) # shape N, (d+1)
    Xap = np.multiply(np.matmul(X, a_reshape.T), p) # shape N, C
    Xapw = np.matmul(Xap, w) # N, (d+1)
    Xap_row_sum = np.sum(Xap, axis=1, keepdims=False) # shape N
    weighted_w = np.matmul(p, w) # shape N, (d+1)
    result = weighted_a.reshape(-1)

    for i in range(N):
        result[i*(d+1): (i+1)*(d+1)] += -a_reshape[y[i]] + Xapw[i] - Xap_row_sum[i] * weighted_w[i]

    result /= N
    return result

