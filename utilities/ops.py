from itertools import product
import numpy as np
import torch
from torch import nn

def clip_grads(params, clip_value):
    if not clip_value > 0:
        return
    for param in params:
        if param.grad is not None:
            param.grad.data.clamp_(-clip_value, clip_value)


def scale_grads(params, threshold):
    if not threshold > 0:
        return
    for param in params:
        l2 = torch.norm(param, 2).data
        if (l2 > threshold).any():
            param.grad.data *= threshold / l2


def get_lr(optimiser):
    for pg in optimiser.param_groups:
        lr = pg['lr']
    return lr


def match_weights(n_layers):
    rnn_fmt = "{}_{}_l{}".format
    cells_fmt = "{}.{}_{}".format

    n = range(n_layers)
    ltype = ['ih', 'hh']
    wtype = ['bias', 'weight']
    matchings = []
    for n, l, w in product(n, ltype, wtype):
        matchings.append((rnn_fmt(w, l, n), cells_fmt(n, w, l)))

    return matchings

def make_safe(x):
    return x.clamp(1e-7, 1 - 1e-7)


def binary_entropy(x):
    return - (x * x.log() + (1 - x) * (1 - x).log())


def info_gain(x):
    marginal = binary_entropy(x.mean(0))
    conditional = binary_entropy(x).mean(0)
    return marginal - conditional


def init_params(m):
    for module_name, module in m.named_modules():
        for param_name, param in module.named_parameters():
            if 'weight' in param_name:
                if 'conv' in param_name or 'lin' in param_name or 'ih' in param_name:
                    nn.init.xavier_uniform_(param)
                elif 'hh' in param_name:
                    nn.init.orthogonal_(param)
            elif param_name == 'bias':
                nn.init.constant_(param, 0.0)


def qfun_loss(y, p):
    log_p = torch.log(make_safe(p))
    positive = torch.sum(log_p, 1)
    neg_prod = torch.exp(positive)
    negative = torch.log1p(-make_safe(neg_prod))
    return - torch.sum(y * positive + (1 - y) * negative)


def kl_anneal_function(anneal_function, step, k1=0.1, k2=0.2, max_value=1.0, x0=100):
    assert anneal_function in ['logistic', 'linear', 'step', 'cyclical'], 'unknown anneal_function'
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(- k1 * (step - x0))))
    elif anneal_function == 'step':
        cnt = step // x0
        step = step % x0
        if cnt > 0:
            max_value -= cnt * 0.1
            max_value = max(0.1, max_value)  
        ma = min(k2 * cnt + k2, max_value)
        mi = 0.01 + k1 * cnt
        return min(ma, mi + 2 * step * (max(ma - mi, 0)) / x0)
    elif anneal_function == 'linear':
        return min(max_value, 0.01 + step / x0)
    elif anneal_function == 'cyclical':
        cnt = step // x0 // 5
        step = step % x0
        ma = min(k2 * cnt + k2, max_value)
        mi = k1
        return min(ma, ma * cnt + mi + 2 * step * (ma - mi) / x0)


class VAELoss(torch.autograd.Function):
    def __init__(self, loss_weight=None):
        if loss_weight is not None:
            loss_weight = torch.FloatTensor(loss_weight).cuda()
        self.softmax_xentropy = nn.CrossEntropyLoss(weight=loss_weight, reduction='sum')
    
    def forward(self, x, x_decoded_mean, z_mean, z_log_var):
        x = x.contiguous().view(-1)
        x_decoded_mean = x_decoded_mean.contiguous().view(-1, x_decoded_mean.size(-1))
        xent_loss = self.softmax_xentropy(input=x_decoded_mean, target=x)
        kl_loss = - 0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return xent_loss, kl_loss


def corresponding(values, idxs, dim=-1):
    if len(values.size()) == 4:
        if len(idxs.size()) == 2:
            idxs = idxs.unsqueeze(0)
        idxs = idxs.repeat(values.size()[0], 1, 1)
    idxs = idxs.unsqueeze(dim)
    res = values.gather(dim, idxs).squeeze(dim)
    return res

def preds2seqs(preds):
    seqs = [torch.cat([torch.multinomial(char_preds, 1)
                       for char_preds in seq_preds])
            for seq_preds in preds]
    return torch.stack(seqs).data


def seqs_equal(seqs1, seqs2):
    return [torch.eq(s1, s2).all() for s1, s2 in zip(seqs1, seqs2)]


def sample(z_mean, z_log_var, size, epsilon_std=1.0):
    epsilon = torch.FloatTensor(*size).normal_(0, epsilon_std).cuda()
    return z_mean + torch.exp(0.5 * z_log_var) * epsilon


# def sample_prior(n_samples, latent_size, decoder, model=None):
#     samples = torch.FloatTensor(n_samples, latent_size).normal_(0, 1)
#     p_hat = decoder.forward(samples)
#     if model:
#         decoded = to_numpy(model.forward_cells(p_hat))
#     else:
#         decoded = decode(p_hat)
#     return decoded


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)