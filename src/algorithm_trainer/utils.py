import subprocess
import numpy as np
import torch
import torch.nn.functional as F


def set_lr(lr, epoch, cycle, start_lr, end_lr):
    if epoch % cycle == 0:
        return start_lr
    return lr + (end_lr - start_lr) / cycle  


def accuracy(preds, y):
    _, preds = torch.max(preds.data, 1)
    total = y.size(0)
    correct = (preds == y).sum().float()
    return (correct / total).item()


def spectral_norm(weight_mat, limit=10., n_power_iterations=2, eps=1e-12, device='cpu'):
    h, w = weight_mat.size()
    # randomly initialize `u` and `v`
    u = F.normalize(torch.randn(h), dim=0, eps=eps).to(device)
    v = F.normalize(torch.randn(w), dim=0, eps=eps).to(device)
    with torch.no_grad():
        for _ in range(n_power_iterations):
            # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
            # are the first left and right singular vectors.
            # This power iteration produces approximations of `u` and `v`.
            v = F.normalize(
                torch.mv(weight_mat.t(), u), dim=0, eps=eps, out=v)
            u = F.normalize(
                torch.mv(weight_mat, v), dim=0, eps=eps, out=u)   
        sigma = torch.dot(u, torch.mv(weight_mat, v))
    if sigma > limit:
        weight_mat = (weight_mat / sigma) * limit
    return weight_mat


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def get_git_revision_hash():
    return str(subprocess.check_output(['git', 'rev-parse', 'HEAD']))



def add_fc(old_model, prefc_feature_sz, num_classes):
    if isinstance(old_model, torch.nn.DataParallel):
        old_model.module.fc = torch.nn.Linear(
            prefc_feature_sz, num_classes).cuda()
        new_model = torch.nn.DataParallel(
            old_model.module, device_ids=range(torch.cuda.device_count()))
    else:
        old_model.fc = torch.nn.Linear(
            prefc_feature_sz, num_classes).cuda()
        new_model = old_model
    return new_model


def remove_fc(model):
    if isinstance(model, torch.nn.DataParallel):
        model.module.fc = None
    else:
        model.fc = None
    return model





def get_swa_model(swa_model, model, alpha=1):
    for param1, param2 in zip(swa_model.parameters(), model.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha
    return swa_model


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))



def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """
    # print(indices)
    encoded_indices = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indices = encoded_indices.scatter_(1,index,1)
    
    return encoded_indices


def smooth_loss(logits, labels, num_classes, eps):
    """compute cross entropy loss using label smoothing

    Args:
        logits (torch Tensor): what is the shape?
        labels (Tensor): integer class labels
        num_classes (int): the total number of classes
        eps (float): the smoothing constant, the probability of eps is split over (num_classes - 1) wrong classes.

    Returns:
        torch Tensor: float of the label-smoothing cross entropy loss
    """
    smoothed_one_hot = one_hot(labels.reshape(-1), num_classes)
    smoothed_one_hot = smoothed_one_hot * (1 - eps) + (1 - smoothed_one_hot) * eps / (num_classes - 1)
    log_prb = F.log_softmax(logits.reshape(-1, num_classes), dim=1)
    loss = -(smoothed_one_hot * log_prb).sum(dim=1)
    # print("loss:", loss)
    loss = loss.mean()
    return loss


def get_labels(y_batch, n_way, n_shot, n_query, batch_sz, mgr_n_query=None, rp=None):
    # original y_batch: (batch_sz*n_way, n_shot+n_query)
    y_batch = y_batch.reshape(batch_sz, n_way, -1)
    # batch_sz, n_way, n_shot+n_query
    
    for i in range(y_batch.shape[0]):
        uniq_classes = np.unique(y_batch[i, :, :].cpu().numpy())
        conversion_dict = {v:k for k, v in enumerate(uniq_classes)}
        # convert labels
        for uniq_class in uniq_classes: 
            y_batch[i, y_batch[i]==uniq_class] = conversion_dict[uniq_class]
        
    shots_y = y_batch[:, :, :n_shot]
    if rp is not None:
        query_y = []
        for c in range(n_way):
            indices = rp[(rp>=(c*2*n_query)) & (rp<((c+1)*2*n_query))] - (c*2*n_query)
            query_y.append(y_batch[:, c, n_shot + indices])
        query_y = torch.cat(query_y, dim=1)
    else:
        query_y = y_batch[:, :, n_shot:]
    shots_y = shots_y.reshape(batch_sz, -1)
    query_y = query_y.reshape(batch_sz, -1)
    return shots_y, query_y


def update_sum_measurements(sum_measurements, measurements):
    for key in measurements.keys():
        sum_measurements[key] += np.sum(measurements[key])

def update_sum_measurements_trajectory(sum_measurements_trajectory, measurements_trajectory):
    for key in measurements_trajectory:
        sum_measurements_trajectory[key] += np.sum(measurements_trajectory[key], axis=0)

def divide_measurements(measurements, n):
    for key in measurements:
        measurements[key] /= n
    return measurements

def average_measurements(measurements):
    # measurements is a dictionary from
    # measurement's name to a list of measurements over the batch of tasks
    avg_measurements = {}
    for key in measurements.keys():
        avg_measurements[key] = torch.mean(measurements[key]).item()
    return avg_measurements

def average_measurements_trajectory(measurements_trajectory):
    avg_measurements_trajectory = {}
    for key in measurements_trajectory:
        avg_measurements_trajectory[key] = np.mean(measurements_trajectory[key], axis=0)
    return avg_measurements_trajectory

def standard_deviation_measurement(measurements):
    std_measurements = {}
    for key in measurements.keys():
        std_measurements[key] = torch.std(measurements[key]).item()
    return std_measurements
