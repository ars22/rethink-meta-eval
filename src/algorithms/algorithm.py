from collections import defaultdict, OrderedDict
import warnings
import time
import numpy as np
import itertools
from itertools import chain

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from src.algorithms.grad import soft_clip, get_grad_norm, get_grad_quantiles
from src.algorithm_trainer.utils import accuracy, spectral_norm
from src.algorithms.utils import one_hot, computeGramMatrix, binv, batched_kronecker, copy_and_replace
from src.algorithms.utils import logistic_regression_hessian_pieces_with_respect_to_w, logistic_regression_hessian_with_respect_to_w, logistic_regression_mixed_derivatives_with_respect_to_w_then_to_X, logistic_regression_mixed_derivatives_with_respect_to_w_then_to_X_left_multiply
from qpth.qp import QPFunction

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter("always", ConvergenceWarning)



class Algorithm(object):
    def __init__(self):
        pass

    def inner_loop_adapt(self, task, num_updates=None, analysis=False, iter=None):
        raise NotImplementedError('Subclass should implement inner_loop')

    def predict_without_adapt(self, train_task, batch, param_dict=None):
        raise NotImplementedError()



class InitBasedAlgorithm(Algorithm):

    def __init__(self, model, loss_func, device, alpha, method, 
            inner_loop_grad_clip, inner_update_method):
        
        self._model = model
        self._device = device
        self._loss_func = loss_func
        self._alpha = alpha # inner loop lr
        self._inner_loop_grad_clip = inner_loop_grad_clip
        self._method = method
        self._second_order = (self._method == 'MAML')
        self._inner_update_method = inner_update_method
        self._beta2 = 0.999
        print("Init based Algorithm: ", self._method)
        print("Init based Algorithm update step type: ", self._inner_update_method)
        self.to(self._device)
        
        

    def get_logits(self, model, X):
        # compute loss on support set
        orig_X_shape = X.shape
        logits = model(
            X.reshape(-1, *orig_X_shape[2:])).reshape(*orig_X_shape[:2], -1)        
        return logits


    def compute_gradient_wrt_model(self, X, y, model, params_wrt_grad_is_computed, create_graph):
        """Compute gradient of self._loss_func(X, y; model),
        based on support, support_labels set but with respect to parameters in model
        """
        
        # compute logits wrt param_dict if param_dict is not None
        logits = self.get_logits(model=model, X=X)
        logits = logits.reshape(-1, logits.size(-1))
        y = y.reshape(-1)
        loss = self._loss_func(logits, y)
        accu = accuracy(logits, y)
        grad_list = torch.autograd.grad(loss, params_wrt_grad_is_computed,
                                    create_graph=create_graph, allow_unused=False, only_inputs=True)
        # allow_unused If False, specifying inputs that were not used when computing outputs
        # (and therefore their grad is always zero) is an error. Defaults to False.
        return loss, accu, grad_list


    def perform_update(self, grad_list):

        if self._inner_update_method == 'sgd':
            return grad_list
        else:
            raise ValueError("inner-method not implemented.")



    def get_updated_model(self, model, grad_list):
        """ model_param = model_param - alpha * grad_list
        """
        updates = []
        updated_grad_list = self.perform_update(grad_list=grad_list)
        for (name, param), grad in zip(model.named_parameters(), updated_grad_list):
            # grad will be torch.Tensor
            assert grad is not None, f"Grad is None for {name}"
            if self._inner_loop_grad_clip > 0:
                grad = grad.clamp(min=-self._inner_loop_grad_clip,
                    max=self._inner_loop_grad_clip)
            update = param - self._alpha * grad
            updates.append(update)
        updates = dict(zip(model.parameters(), updates))
        do_not_copy = [tensor for tensor in chain(model.parameters(), model.buffers())
                if tensor not in updates]
        return copy_and_replace(model, updates, do_not_copy=do_not_copy)
                


    def get_param_diff(self, params_1, params_2):
        """ returns params_1 - params_2
        """
        diff = []
        for param_1, param_2 in zip(params_1, params_2):
            diff.append(param_1 - param_2)
        return diff


    def populate_grad(self, grad_list):
        """Take values in grad list and populate param.grad with it
        for param in model parameters.
        """
        for param, calculated_grad in zip(self._model.parameters(), grad_list):
            if param.grad is not None: 
                param.grad += calculated_grad.detach()
            else:
                param.grad = calculated_grad.detach()


    def inner_loop_adapt(self, support, support_labels, query, query_labels,
        n_way, n_shot, n_query, num_updates_inner):

        
        # adapt means doing the complete inner loop update
        measurements_trajectory = defaultdict(list)
        # copy every tenso's data in the original dictionary
        updated_model = self._model         
        
        assert num_updates_inner > 0
        for i in range(num_updates_inner):
            support_loss, support_accu, grad_list = self.compute_gradient_wrt_model(
                X=support, y=support_labels, model=updated_model, params_wrt_grad_is_computed=updated_model.parameters(),
                create_graph=self._second_order)
            updated_model = self.get_updated_model(model=updated_model, 
                grad_list=grad_list)
            

        # Now compute loss on query set and from that the outer gradient
        if self._method == 'MAML':
            query_loss, query_accu, outer_grad_list = self.compute_gradient_wrt_model(
                X=query, y=query_labels, model=updated_model, params_wrt_grad_is_computed=self._model.parameters(),
                create_graph=False)
        elif self._method == 'FOMAML':
            query_loss, query_accu, outer_grad_list = self.compute_gradient_wrt_model(
                X=query, y=query_labels, model=updated_model, params_wrt_grad_is_computed=updated_model.parameters(),
                create_graph=False)
        elif self._method == 'Reptile': 
            query_loss, query_accu, grad_list = self.compute_gradient_wrt_model(
                X=query, y=query_labels, model=updated_model, params_wrt_grad_is_computed=updated_model.parameters(),
                create_graph=False)
            updated_model = self.get_updated_model(model=updated_model, 
                grad_list=grad_list)
            outer_grad_list = self.get_param_diff(self._model.parameters(), updated_model.parameters())
        else:
            raise ValueError("Meta-alg not implemented.")
            
        # populate model.grad with outer_grad_list
        self.populate_grad(outer_grad_list)
        
        # metrics
        measurements_trajectory['loss'].append(support_loss.item())
        measurements_trajectory['accu'].append(support_accu * 100.)
        measurements_trajectory['mt_outer_loss'].append(query_loss.item())
        measurements_trajectory['mt_outer_accu'].append(query_accu * 100.)
        return measurements_trajectory


    def to(self, device, **kwargs):
        self._device = device
        self._model.to(device, **kwargs)


    def state_dict(self):
        # for model saving and reloading
        return {'model': self._model.state_dict()}





class SVM(Algorithm):

    def __init__(self, model, inner_loss_func, device, scale,
        C_reg=0.1, max_iter=15, double_precision=False):
        
        self._model = model
        self._device = device
        self._inner_loss_func = inner_loss_func
        self._C_reg = C_reg
        self._max_iter = max_iter
        self._double_precision = double_precision
        self._scale = scale
        self.to(self._device)

        # scale
        if isinstance(model, torch.nn.DataParallel) and hasattr(model.module, 'scale'):
            self._scale = model.module.scale
        elif hasattr(model, 'scale'):
            self._scale = model.scale
        print("Algorithm logits scale:", self._scale)


    def inner_loop_adapt(self, support, support_labels, query, n_way, n_shot, n_query):
        """
        Fits the support set with multi-class SVM and 
        returns the classification score on the query set.
        
        This is the multi-class SVM presented in:
        On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines
        (Crammer and Singer, Journal of Machine Learning Research 2001).
        This model is the classification head that we use for the final version.
        Parameters:
        query:  a (tasks_per_batch, n_query, c, h, w) Tensor.
        support:  a (tasks_per_batch, n_support, c, h, w) Tensor.
        support_labels: a (tasks_per_batch, n_support) Tensor.
        n_way: a scalar. Represents the number of classes in a few-shot classification task.
        n_shot: a scalar. Represents the number of support examples given per class.
        C_reg: a scalar. Represents the cost parameter C in SVM.
        Returns: a (tasks_per_batch, n_query, n_way) Tensor.
        """

        measurements_trajectory = defaultdict(list)

        assert(query.dim() == 5)
        assert(support.dim() == 5)
        
        # get features
        orig_query_shape = query.shape
        orig_support_shape = support.shape
        support = self._model(
            support.reshape(-1, *orig_support_shape[2:]), only_features=True).reshape(*orig_support_shape[:2], -1)
        query = self._model(
            query.reshape(-1, *orig_query_shape[2:]), only_features=True).reshape(*orig_query_shape[:2], -1)
                

        tasks_per_batch = query.size(0)
        total_n_support = support.size(1) # support samples across all classes in a task
        total_n_query = query.size(1)     # query samples across all classes in a task
        d = query.size(2)                 # dimension

        C_reg = self._C_reg
        maxIter = self._max_iter

        assert(query.dim() == 3)
        assert(support.dim() == 3)
        assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
        assert(total_n_support == n_way * n_shot)      # total_n_support must equal to n_way * n_shot
        assert(total_n_query == n_way * n_query)      # total_n_query must equal to n_way * n_query


        #Here we solve the dual problem:
        #Note that the classes are indexed by m & samples are indexed by i.
        #min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
        #s.t.  \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i

        #where w_m(\alpha) = \sum_i \alpha^m_i x_i,
        #and C^m_i = C if m  = y_i,
        #C^m_i = 0 if m != y_i.
        #This borrows the notation of liblinear.
        
        #\alpha is an (total_n_support, n_way) matrix
        kernel_matrix = computeGramMatrix(support, support)

        id_matrix_0 = torch.eye(n_way).expand(tasks_per_batch, n_way, n_way).cuda()
        block_kernel_matrix = batched_kronecker(kernel_matrix, id_matrix_0)
        #This seems to help avoid PSD error from the QP solver.
        block_kernel_matrix += 1.0 * torch.eye(n_way*total_n_support).expand(
            tasks_per_batch, n_way*total_n_support, n_way*total_n_support).cuda()
        
        support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * total_n_support), n_way) 
        # (tasks_per_batch * total_n_support, n_way)
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, total_n_support, n_way)
        support_labels_one_hot = support_labels_one_hot.reshape(tasks_per_batch, total_n_support * n_way)
        # (tasks_per_batch, total_n_support * n_way)

        G = block_kernel_matrix
        e = -1.0 * support_labels_one_hot
        #This part is for the inequality constraints:
        #\alpha^m_i <= C^m_i \forall m,i
        #where C^m_i = C if m  = y_i,
        #C^m_i = 0 if m != y_i.
        id_matrix_1 = torch.eye(n_way * total_n_support).expand(tasks_per_batch, n_way * total_n_support, n_way * total_n_support)
        C = Variable(id_matrix_1)
        h = Variable(C_reg * support_labels_one_hot)
        #print (C.size(), h.size())
        #This part is for the equality constraints:
        #\sum_m \alpha^m_i=0 \forall i
        id_matrix_2 = torch.eye(total_n_support).expand(tasks_per_batch, total_n_support, total_n_support).cuda()

        A = Variable(batched_kronecker(id_matrix_2, torch.ones(tasks_per_batch, 1, n_way).cuda()))
        b = Variable(torch.zeros(tasks_per_batch, total_n_support))

        if self._double_precision:
            G, e, C, h, A, b = [x.double().cuda() for x in [G, e, C, h, A, b]]
        else:
            G, e, C, h, A, b = [x.float().cuda() for x in [G, e, C, h, A, b]]

        # Solve the following QP to fit SVM:
        #        \hat z =   argmin_z 1/2 z^T G z + e^T z
        #                 subject to Cz <= h
        # We use detach() to prevent backpropagation to fixed variables.
        qp_sol = QPFunction(verbose=False, maxIter=maxIter)(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach())
        # G is not detached, that is the only one that needs gradients, since its a function of phi(x).

        qp_sol = qp_sol.reshape(tasks_per_batch, total_n_support, n_way)

        
        # Compute the classification score for query.
        compatibility_query = computeGramMatrix(support, query)
        compatibility_query = compatibility_query.float()
        compatibility_query = compatibility_query.unsqueeze(3).expand(tasks_per_batch, total_n_support, total_n_query, n_way)
        logits_query = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, total_n_support, total_n_query, n_way)
        logits_query = logits_query * compatibility_query
        logits_query = torch.sum(logits_query, 1) * self._scale

        # Compute the classification score for support.
        with torch.no_grad():
            compatibility_support = computeGramMatrix(support, support)
            compatibility_support = compatibility_support.float()
            compatibility_support = compatibility_support.unsqueeze(3).expand(tasks_per_batch, total_n_support, total_n_support, n_way)
            logits_support = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, total_n_support, total_n_support, n_way)
            logits_support = logits_support * compatibility_support
            logits_support = torch.sum(logits_support, 1) * self._scale
            
        # compute loss and acc on support
        logits_support = logits_support.reshape(-1, logits_support.size(-1))
        labels_support = support_labels.reshape(-1)
        
        loss = self._inner_loss_func(logits_support, labels_support)
        accu = accuracy(logits_support, labels_support)
        measurements_trajectory['loss'].append(loss.item())
        measurements_trajectory['accu'].append(accu)


        return logits_query, measurements_trajectory

    
    def to(self, device, **kwargs):
        self._device = device
        self._model.to(device, **kwargs)
        

    def state_dict(self):
        # for model saving and reloading
        return {'model': self._model.state_dict()}



class ProtoNet(Algorithm):

    def __init__(self, model, inner_loss_func, device, 
             metric, scale, normalize=True):
        
        self._model = model
        self._device = device
        self._inner_loss_func = inner_loss_func
        self._normalize = normalize
        self._scale = scale
        self._metric = metric # euc or cos
        self.to(self._device)
        
        # scale
        if isinstance(model, torch.nn.DataParallel) and hasattr(model.module, 'scale'):
            self._scale = model.module.scale
        elif hasattr(model, 'scale'):
            self._scale = model.scale
        print("Algorithm logits scale:", self._scale)

   
    def inner_loop_adapt(self, support, support_labels, query, n_way, n_shot, n_query):
        """
        Constructs the prototype representation of each class(=mean of support vectors of each class) and 
        returns the classification score (=L2 distance to each class prototype) on the query set.
        
        This model is the classification head described in:
        Prototypical Networks for Few-shot Learning
        (Snell et al., NIPS 2017).
        
        Parameters:
        query:  a (n_tasks_per_batch, n_query, c, h, w) Tensor.
        support:  a (n_tasks_per_batch, n_support, c, h, w) Tensor.
        support_labels: a (n_tasks_per_batch, n_support) Tensor.
        n_way: a scalar. Represents the number of classes in a few-shot classification task.
        n_shot: a scalar. Represents the number of support examples given per class.
        normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
        Returns: a (tasks_per_batch, n_query, n_way) Tensor.
        """

        measurements_trajectory = defaultdict(list)

        assert(query.dim() == 5)
        assert(support.dim() == 5)
        
        # get features
        orig_query_shape = query.shape
        orig_support_shape = support.shape

        support = self._model(
            support.reshape(-1, *orig_support_shape[2:]), only_features=True).reshape(*orig_support_shape[:2], -1)
        query = self._model(
            query.reshape(-1, *orig_query_shape[2:]), only_features=True).reshape(*orig_query_shape[:2], -1)
        

        tasks_per_batch = query.size(0)
        total_n_support = support.size(1) # support samples across all classes in a task
        total_n_query = query.size(1)     # query samples across all classes in a task
        d = query.size(2)                 # dimension
        
        assert(query.dim() == 3)
        assert(support.dim() == 3)
        assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
        assert(total_n_support == n_way * n_shot)
        assert(total_n_query == n_way * n_query)

        support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * total_n_support), n_way)
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, total_n_support, n_way)
    
        labels_train_transposed = support_labels_one_hot.transpose(1,2)
        # this makes it tasks_per_batch x n_way x total_n_support

        prototypes = torch.bmm(labels_train_transposed, support)
        # [batch_size x n_way x d] =
        #     [batch_size x n_way x total_n_support] * [batch_size x total_n_support x d]

        prototypes = prototypes.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
        )
        # Divide with the number of examples per novel category.

        
        if self._metric == 'euclidean':

            ################################################
            # Compute the classification score for query
            ################################################

            # Distance Matrix Vectorization Trick
            AB = computeGramMatrix(query, prototypes)
            # batch_size x total_n_query x n_way
            AA = (query * query).sum(dim=2, keepdim=True)
            # batch_size x total_n_query x 1
            BB = (prototypes * prototypes).sum(dim=2, keepdim=True).reshape(tasks_per_batch, 1, n_way)
            # batch_size x 1 x n_way
            logits_query = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)
            # euclidean distance 
            logits_query = -logits_query
            # batch_size x total_n_query x n_way
            logits_query = logits_query * self._scale

            if self._normalize:
                logits_query = logits_query / d
            
            ################################################
            # Compute the classification score for support
            ################################################

            # Distance Matrix Vectorization Trick
            AB = computeGramMatrix(support, prototypes)
            # batch_size x total_n_support x n_way
            AA = (support * support).sum(dim=2, keepdim=True)
            # batch_size x total_n_support x 1
            ## BB needn't be computed again
            logits_support = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)
            # euclidean distance 
            logits_support = -logits_support
            # batch_size x total_n_support x n_way
            logits_support = logits_support * self._scale

            if self._normalize:
                logits_support = logits_support / d
        
        elif self._metric == 'cosine':
            
            ################################################
            # Compute the classification score for query
            ################################################

            # Distance Matrix Vectorization Trick
            logits_query = computeGramMatrix(query, prototypes)
            logits_query = logits_query * self._scale
            # batch_size x total_n_query x n_way
        
            ################################################
            # Compute the classification score for support
            ################################################

            # Distance Matrix Vectorization Trick
            with torch.no_grad():
                logits_support = computeGramMatrix(support, prototypes)
                logits_support = logits_support * self._scale
                # batch_size x total_n_support x n_way


        else:
            raise ValueError("Metric not implemented")

        
        # compute loss and acc on support
        logits_support = logits_support.reshape(-1, logits_support.size(-1))
        labels_support = support_labels.reshape(-1)
        loss = self._inner_loss_func(logits_support, labels_support)
        accu = accuracy(logits_support, labels_support) * 100.
        
        # logging
        measurements_trajectory['loss'].append(loss.item())
        measurements_trajectory['accu'].append(accu)

        return logits_query, measurements_trajectory


    def to(self, device, **kwargs):
        self._device = device
        self._model.to(device, **kwargs)

    def state_dict(self):
        # for model saving and reloading
        return {'model': self._model.state_dict()}








class Ridge(Algorithm):

    def __init__(self, model, inner_loss_func, device, 
            scale, normalize=True):
        
        self._model = model
        self._device = device
        self._inner_loss_func = inner_loss_func
        self._normalize = normalize
        self._lambda_reg = 50.0
        self._double_precision = False
        self._scale = scale
        self.to(self._device)

        # scale
        if isinstance(model, torch.nn.DataParallel) and hasattr(model.module, 'scale'):
            self._scale = model.module.scale
        elif hasattr(model, 'scale'):
            self._scale = model.scale
        print("Algorithm logits scale:", self._scale)



    def inner_loop_adapt(self, support, support_labels, query, n_way, n_shot, n_query):

        """
        Fits the support set with ridge regression and 
        returns the classification score on the query set.
        Parameters:
        query:  a (n_tasks_per_batch, n_query, c, h, w) Tensor.
        support:  a (n_tasks_per_batch, n_support, c, h, w) Tensor.
        support_labels: a (tasks_per_batch, n_support) Tensor.
        lambda_reg: a scalar. Represents the strength of L2 regularization.
        Returns: a (tasks_per_batch, n_query, n_way) Tensor.
        """


        measurements_trajectory = defaultdict(list)

        assert(query.dim() == 5)
        assert(support.dim() == 5)
        
        # get features
        orig_query_shape = query.shape
        orig_support_shape = support.shape
        
        support = self._model(
            support.reshape(-1, *orig_support_shape[2:]), only_features=True).reshape(*orig_support_shape[:2], -1)
        query = self._model(
            query.reshape(-1, *orig_query_shape[2:]), only_features=True).reshape(*orig_query_shape[:2], -1)
        
        
        lambda_reg = self._lambda_reg
        double_precision = self._double_precision
        tasks_per_batch = query.size(0)
        total_n_support = support.size(1) # support samples across all classes in a task
        total_n_query = query.size(1)     # query samples across all classes in a task
        d = query.size(2)                 # dimension
        
        assert(query.dim() == 3)
        assert(support.dim() == 3)
        assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
        assert(total_n_support == n_way * n_shot)      # total_n_support must equal to n_way * n_shot
        assert(total_n_query == n_way * n_query)      # total_n_support must equal to n_way * n_shot

        #Here we solve the dual problem:
        #Note that the classes are indexed by m & samples are indexed by i.
        #min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i

        #where w_m(\alpha) = \sum_i \alpha^m_i x_i,
        
        #\alpha is an (total_n_support, n_way) matrix
        kernel_matrix = computeGramMatrix(support, support)
        kernel_matrix += lambda_reg * torch.eye(total_n_support).expand(tasks_per_batch, total_n_support, total_n_support).cuda()

        block_kernel_matrix = kernel_matrix.repeat(n_way, 1, 1) #(n_way * tasks_per_batch, total_n_support, total_n_support)
        
        support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * total_n_support), n_way) # (tasks_per_batch * total_n_support, n_way)
        support_labels_one_hot = support_labels_one_hot.transpose(0, 1) # (n_way, tasks_per_batch * total_n_support)
        support_labels_one_hot = support_labels_one_hot.reshape(n_way * tasks_per_batch, total_n_support)     # (n_way*tasks_per_batch, total_n_support)
        
        G = block_kernel_matrix
        e = -2.0 * support_labels_one_hot
        
        #This is a fake inequlity constraint as qpth does not support QP without an inequality constraint.
        id_matrix_1 = torch.zeros(tasks_per_batch*n_way, total_n_support, total_n_support)
        C = Variable(id_matrix_1)
        h = Variable(torch.zeros((tasks_per_batch*n_way, total_n_support)))
        dummy = Variable(torch.Tensor()).cuda()      # We want to ignore the equality constraint.

        if double_precision:
            G, e, C, h = [x.double().cuda() for x in [G, e, C, h]]

        else:
            G, e, C, h = [x.float().cuda() for x in [G, e, C, h]]

        # Solve the following QP to fit SVM:
        #        \hat z =   argmin_z 1/2 z^T G z + e^T z
        #                 subject to Cz <= h
        # We use detach() to prevent backpropagation to fixed variables.
        qp_sol = QPFunction(verbose=False)(G, e.detach(), C.detach(), h.detach(), dummy.detach(), dummy.detach())
        #qp_sol = QPFunction(verbose=False)(G, e.detach(), dummy.detach(), dummy.detach(), dummy.detach(), dummy.detach())

        #qp_sol (n_way*tasks_per_batch, total_n_support)
        qp_sol = qp_sol.reshape(n_way, tasks_per_batch, total_n_support)
        #qp_sol (n_way, tasks_per_batch, total_n_support)
        qp_sol = qp_sol.permute(1, 2, 0)
        #qp_sol (tasks_per_batch, total_n_support, n_way)
        
        # Compute the classification score.
        compatibility = computeGramMatrix(support, query)
        compatibility = compatibility.float()
        compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, total_n_support, total_n_query, n_way)
        qp_sol = qp_sol.reshape(tasks_per_batch, total_n_support, n_way)
        logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, total_n_support, total_n_query, n_way)
        logits = logits * compatibility
        logits = torch.sum(logits, 1) * self._scale

        # compute loss and acc on support
        with torch.no_grad():
            compatibility = computeGramMatrix(support, support)
            compatibility = compatibility.float()
            compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, total_n_support, total_n_support, n_way)
            logits_support = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, total_n_support, total_n_support, n_way)
            logits_support = logits_support * compatibility
            logits_support = torch.sum(logits_support, 1)
            logits_support = logits_support.reshape(-1, logits_support.size(-1)) * self._scale
            loss = self._inner_loss_func(logits_support, support_labels.reshape(-1))
            accu = accuracy(logits_support, support_labels.reshape(-1)) * 100.
            measurements_trajectory['loss'].append(loss.item())
            measurements_trajectory['accu'].append(accu)

        return logits, measurements_trajectory

    def to(self, device, **kwargs):
        self._device = device
        self._model.to(device, **kwargs)

    def state_dict(self):
        # for model saving and reloading
        return {'model': self._model.state_dict()}
