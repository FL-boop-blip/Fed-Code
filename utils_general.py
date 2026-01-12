import copy
import math
from copy import deepcopy

import numpy as np
import torch
from sympy.physics.units.definitions.unit_definitions import gauss
from torchvision.transforms.v2.functional import gaussian_blur

from optimizer.Adan import Adan
from optimizer.sam import SAM, disable_running_stats
from optimizer.utils import ProportionScheduler
from utils_libs import *
from Data.utils_dataset import *
from Model.utils_models import *
from optimizer.ESAM import ESAM
from optimizer.TSAM import TSAM
from optimizer.DRegSAM import DRegSAM
from optimizer.LESAM_D import LESAM_D
from scipy import special
from collections import defaultdict, OrderedDict

# Global parameters
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# from torch.utils.tensorboard import SummaryWriter

import time

max_norm = 10


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = nn.KLDivLoss(reduction='batchmean')(p_s, p_t) * (self.T ** 2)
        return loss
def gaussian_blur(tensor, sigma=0.5, kernel_size=None):
    """
    对4D张量 [B,C,H,W] 应用高斯模糊（兼容CIFAR10_LeNet的输出形状）
    修改点：
    1. 自动计算kernel_size（若未指定）
    2. 使用向量化操作替代逐通道循环
    3. 修复维度不匹配问题
    """
    if sigma <= 0 or tensor.dim() != 4:
        return tensor  # 仅处理4D输入

    # 自动计算核大小
    kernel_size = kernel_size or int(2 * 3 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1  # 确保为奇数

    # 生成1D高斯核（归一化）
    coords = torch.arange(kernel_size, device=tensor.device) - kernel_size // 2
    kernel_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    # 转换为2D高斯核 [1,1,k,k]
    kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, kernel_size, kernel_size)

    # 对每个通道独立卷积（groups=C）
    return F.conv2d(
        tensor,
        kernel_2d.repeat(tensor.size(1), 1, 1, 1),  # [C,1,k,k]
        padding=kernel_size // 2,
        groups=tensor.size(1)
    )


def compute_client_gradient(global_model,trn_x,trn_y,batch_size,dataset_name):
    total_gradient = None
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    trn_gen_iter = trn_gen.__iter__()
    for i in range(int(np.ceil(n_trn / batch_size))):
        batch_x, batch_y = trn_gen_iter.__next__()
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        y_pred = global_model(batch_x)
        loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
        loss = loss_f_i
        global_model.zero_grad()
        loss.backward()
        current_grad = np.concatenate([
            param.grad.detach().cpu().numpy().flatten() for param in global_model.parameters()
        ])

        if total_gradient is None:
            total_gradient = current_grad.copy()
        else:
            total_gradient += current_grad

    return total_gradient
def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_flat_params(model, flat_params):
    pointer = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[pointer:pointer + numel].view_as(p))
        pointer += numel
def compute_D(clnt_gradients,weight_list,n_clnt):
    g_gradient = np.zeros_like(clnt_gradients[0])
    for i in range(len(clnt_gradients)):
        g_gradient += 1/n_clnt * clnt_gradients[i]
    total_gradient_norm = np.linalg.norm(g_gradient, 2)**2
    print('total_gradient_norm is {}'.format(total_gradient_norm))
    normalized_sums = 0.0
    for i, grad in enumerate(clnt_gradients):
        grad_norm_squared = np.linalg.norm(grad, 2)**2
        print('client {}, gradient_norm is {}'.format(i,grad_norm_squared))
        print("weight list {} is {}".format(i,weight_list[i]))
        normalized_sums += 1/n_clnt * (grad_norm_squared / total_gradient_norm)
    return np.sqrt(normalized_sums)



def sinp(epoch,t):
    return 0.5 * (1+ math.sin(math.pi * t / epoch))



def get_update_params(model):
    # model parameters ---> vector (different storage)
    vec = []
    for param in model.parameters():
        vec.append(param.clone().detach().cpu().reshape(-1))
    return torch.cat(vec)


def get_params_list_with_shape(model, param_list, device):
    vec_with_shape = []
    idx = 0
    for param in model.parameters():
        length = param.numel()
        vec_with_shape.append(param_list[idx:idx + length].reshape(param.shape).to(device))
    return vec_with_shape


def get_mdl_params(model):
    # model parameters ---> vector (different storage)
    vec = []
    for param in model.parameters():
        vec.append(param.clone().detach().cpu().reshape(-1))
    return torch.cat(vec)


def param_to_vector(model):
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append(param.reshape(-1))
    return torch.cat(vec)


def get_distribution_difference(client_cls_counts, participation_clients, metric, hypo_distribution):
    local_distributions = client_cls_counts[np.array(participation_clients), :]
    local_distributions = local_distributions / local_distributions.sum(axis=1)[:, np.newaxis]

    if metric == 'cosine':
        similarity_scores = local_distributions.dot(hypo_distribution) / (
                np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(hypo_distribution))
        difference = 1.0 - similarity_scores
    elif metric == 'only_iid':
        similarity_scores = local_distributions.dot(hypo_distribution) / (
                np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(hypo_distribution))
        difference = np.where(similarity_scores > 0.9, 0.01, float('inf'))
    elif metric == 'l1':
        difference = np.linalg.norm(local_distributions - hypo_distribution, ord=1, axis=1)
    elif metric == 'l2':
        difference = np.linalg.norm(local_distributions - hypo_distribution, axis=1)
    elif metric == 'kl':
        difference = special.kl_div(local_distributions, hypo_distribution)
        difference = np.sum(difference, axis=1)

        difference = np.array([0 for _ in range(len(difference))]) if np.sum(difference) == 0 else difference / np.sum(
            difference)
    return difference


def disco_weight_adjusting(old_weight, distribution_difference, a, b):
    weight_tmp = old_weight - a * distribution_difference + b

    if np.sum(weight_tmp > 0) > 0:
        new_weight = np.copy(weight_tmp)
        new_weight[new_weight < 0.0] = 0.0
    else:
        new_weight = np.copy(old_weight)

    total_normalizer = sum([new_weight[r] for r in range(len(old_weight))])
    new_weight = [new_weight[r] / total_normalizer for r in range(len(old_weight))]
    return new_weight


# --- Evaluate a NN model
def get_acc_loss(data_x, data_y, model, dataset_name, w_decay=None):
    acc_overall = 0;
    loss_overall = 0;
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    # batch_size = min(6000, data_x.shape[0])
    batch_size = min(2000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model.eval();
    model = model.to(device)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)

            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_overall += loss.item()

            # Accuracy calculation
            y_pred = y_pred.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct

    loss_overall /= n_tst
    if w_decay != None:
        # Add L2 loss
        params = get_mdl_params([model], n_par=None)
        loss_overall += w_decay / 2 * np.sum(params * params)

    model.train()
    return loss_overall, acc_overall / n_tst


# --- Helper functions

def avg_models(mdl, clnt_models, weight_list):
    n_node = len(clnt_models)
    dict_list = list(range(n_node));
    for i in range(n_node):
        dict_list[i] = copy.deepcopy(dict(clnt_models[i].named_parameters()))

    param_0 = clnt_models[0].named_parameters()

    for name, param in param_0:
        param_ = weight_list[0] * param.data
        for i in list(range(1, n_node)):
            param_ = param_ + weight_list[i] * dict_list[i][name].data
        dict_list[0][name].data.copy_(param_)

    mdl.load_state_dict(dict_list[0])

    # Remove dict_list from memory
    del dict_list

    return mdl



def set_client_from_params(mdl, params):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx + length].reshape(weights.shape)).to(device))
        idx += length

    mdl.load_state_dict(dict_param)
    return mdl


def get_mdl_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)




def eAM(e,r):
    return math.exp(-e * r)

def topk_sparsify(tensor, k_ratio=0.01):
    """
    保留张量中绝对值最大的前k%元素，其余置零
    参数:
        tensor: 待稀疏化的张量
        k_ratio: 保留元素的比例（如0.01表示1%）
    """
    if k_ratio >= 1.0:
        return tensor

    total_elements = tensor.numel()
    k = max(1, int(total_elements * k_ratio))

    # 获取topk索引
    values, indices = torch.topk(tensor.abs().view(-1), k)

    # 构建稀疏掩码
    mask = torch.zeros_like(tensor).view(-1)
    mask[indices] = 1
    mask = mask.view(tensor.shape)

    return tensor * mask








