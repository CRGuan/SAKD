import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy
import math


def KD(y, labels):
    l_ce = F.cross_entropy(y, labels)
    return l_ce


class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss


def distillation_loss(source, target):
    p = F.normalize(source.view(source.size(0), -1))
    q = F.normalize(target.view(source.size(0), -1))
    return (p - q).pow(2).mean()


def self_attention_channel(feat):
    g = feat.pow(2).mean(1)
    q = g
    k = g
    v = g
    t = torch.tensor(math.sqrt(q.shape[2]))
    feats = g + ((nn.Softmax(dim=1)(torch.bmm(q, k.permute(0, 2, 1))) * v) + (nn.Softmax(dim=2)(torch.bmm(q, k.permute(0, 2, 1))) * v) )/ t
    return feats


def self_attention_reshape(feat):
    g = feat.view(256, feat.size(1), -1)
    q = g
    k = g
    v = g
    # t = torch.tensor(math.sqrt(q.shape[2]))
    m = F.normalize(nn.Softmax(dim=1)(torch.bmm(q, k.permute(0, 2, 1))) @ v)
    feats = g + m
    return feats


class Distiller(nn.Module):
    def __init__(self, t_net, s_net):
        super(Distiller, self).__init__()

        self.t_net = t_net
        self.s_net = s_net

    def forward(self, x):

        t_feats, t_out = self.t_net.extract_feature(x, preReLU=False)
        s_feats, s_out = self.s_net.extract_feature(x, preReLU=False)
        feat_num = len(t_feats)
        loss_distill = 0
        for i in range(feat_num):
            t_f_c = self_attention_channel(t_feats[i])
            s_f_c = self_attention_channel(s_feats[i])
            loss_distill += distillation_loss(s_f_c, t_f_c.detach()) / 2 ** (feat_num - i - 1)
        return s_out, t_out, loss_distill

