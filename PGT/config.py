from torch import nn
from transformers import AutoTokenizer, AutoModel
import argparse
import math
import torch
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='MTL Classifier')

parser.add_argument('--epoch', '-e', type=int, default=100, help='maximum iteration to train (default: 100)')
parser.add_argument('--lr', '-l', type=float, default=5e-6, help='learning rates for steps')
parser.add_argument('--weight_decay','-w', type=float, default=1e-2)
parser.add_argument('--scale','-s', type=float, default=20, help="ArcFace's sphere scaling")
parser.add_argument('--margin','-m', type=float, default=0.0, help="ArcFace's margin")
args = parser.parse_args()

class Config(object):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    epoch = args.epoch
    lr = args.lr
    weight_decay = args.weight_decay
    scale = args.scale
    margin = args.margin

    batch = {
        "ATIS": 10,
        "BANKING77" :10,
        "clinc150":14,
        "HWU64":10,
        "mcid":10,
        "restaurant":10
    }
    
    num_labels = {
            "ATIS": 25,
            "BANKING77": 77,
            "clinc150": 150,
            "HWU64": 64,
            "mcid": 16,
            "restaurant": 13
    }
    
    dropout_prob = 0.1





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