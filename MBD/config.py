import sys
from transformers import AutoTokenizer, AutoModel
from transformers import logging
import argparse

logging.set_verbosity_error()


parser = argparse.ArgumentParser(description='MTL Classifier')

parser.add_argument('--epoch', '-e', type=int, default=100, help='maximum iteration to train (default: 100)')
parser.add_argument('--lr', '-l', type=float, default=5e-6, help='learning rates for steps')
parser.add_argument('--weight_decay','-w', type=float, default=1e-2)
parser.add_argument('--scale','-s', type=float, default=20, help="ArcFace's sphere scaling")
parser.add_argument('--margin','-m', type=float, default=0.0, help="ArcFace's margin")
args = parser.parse_args()

config = {}
class Config(object):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    # model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

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
