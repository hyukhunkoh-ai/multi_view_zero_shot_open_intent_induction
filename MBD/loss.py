import math
import torch
from torch import nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    def __init__(self, batch_size=32, in_feature=128, out_feature=2, s=1.0, m=0):
        super(ArcFace, self).__init__()

        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.batch_size = batch_size

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    # bs = 64 (ab x 32, n x 32)
    def forward(self, x, label_onehot):
        
        # cos(theta)
        normalized_x = F.normalize(x, dim=-1)
        cos_theta = F.linear(normalized_x, F.normalize(self.weight))

        # cos(theta + m)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_phi = cos_theta * self.cos_m - sin_theta * self.sin_m
        
        cos_phi = (label_onehot * cos_phi) + ((1 - label_onehot) * cos_theta)
        cos_phi = self.s * cos_phi

        return cos_phi