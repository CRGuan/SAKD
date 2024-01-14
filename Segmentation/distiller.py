import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy

import math



class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='mean') * (self.T**2) / y_s.shape[0]
        return loss


def distillation_loss(source, target):
    p = F.normalize(source.view(source.size(0), -1))
    q = F.normalize(target.view(source.size(0), -1))
    return (p - q).pow(2).mean()


def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)


def self_attention_channel(feat):
    g = feat.pow(2).mean(1)
    q = g
    k = g
    v = g
    t = torch.tensor(math.sqrt(q.shape[2]))
    feats = g + ((nn.Softmax(dim=1)(torch.bmm(q, k.permute(0, 2, 1))) * v) + (nn.Softmax(dim=2)(torch.bmm(q, k.permute(0, 2, 1))) * v) )/ t
    return feats


class Distiller(nn.Module):
    def __init__(self, t_net, s_net):
        super(Distiller, self).__init__()

        t_channels = t_net.get_channel_num()
        s_channels = s_net.get_channel_num()

        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        self.t_net = t_net
        self.s_net = s_net

        self.loss_divider = [8, 4, 2, 1, 1, 4*4]

    def forward(self, x):

        t_feats, t_out = self.t_net.extract_feature(x)
        s_feats, s_out = self.s_net.extract_feature(x)
        feat_num = len(t_feats)

        loss_distill = 0

        for i in range(feat_num):
            s_feats[i] = self.Connectors[i](s_feats[i])
            t_f_c = self_attention_channel(t_feats[i])
            s_f_c = self_attention_channel(s_feats[i])
            loss_distill += distillation_loss(s_f_c, t_f_c.detach()) / self.loss_divider[i]

        return s_out, t_out, loss_distill
