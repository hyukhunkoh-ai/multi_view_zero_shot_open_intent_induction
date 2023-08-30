import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gc
import random
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
import torch.backends.cudnn as cudnn
from datasets import load_metric
from transformers import logging
from transformers import get_scheduler
from tqdm.auto import tqdm
from datetime import datetime

from MBD.train import train
from MBD.validation import validation
from MBD.config import Config
from MBD.MBD_Model import MBD_Model
from MBD.utils import (make_arcface, make_dataloader)


if __name__ == '__main__':

    SEED = 22
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(SEED)

    """
    ATIS        5870        3.36        3
    BANKING77   13072       7.49        7
    clinc150    22500       12.89       13
    HWU64       11033       6.32        6
    mcid        1745        1           1
    restaurant  4180        2.39        2
    """

    logging.set_verbosity_error()
    gc.collect()
    torch.cuda.empty_cache()    

    # Make Directories 
    data_list = ['ATIS', 'BANKING77', 'clinc150','HWU64','mcid', 'restaurant']

    if not os.path.exists(f'./results_{Config.lr}_{Config.weight_decay}_{Config.scale}_{Config.margin}/'):
        for data in data_list:
            os.makedirs(f'./results_{Config.lr}_{Config.weight_decay}_{Config.scale}_{Config.margin}/{data}/')

    print(f"==================== results_{Config.lr}_{Config.weight_decay}_{Config.scale}_{Config.margin} Created ====================")

    PATH = os.path.join(os.getcwd(), f'./results_{Config.lr}_{Config.weight_decay}_{Config.scale}_{Config.margin}/')
    auc_file = os.path.join(PATH, 'total_AUC.txt')
    loss_file = os.path.join(PATH, 'total_loss.txt')
    auc_data_file = os.path.join(PATH, 'AUC_DATA.txt')

    log_dir = os.path.join(PATH, 'logs/')

    # Print Device Info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device:{device}')

    if torch.cuda.is_available():
        print("Current cuda device: ", torch.cuda.current_device())
        print("Count of Using GPUs: ", torch.cuda.device_count())

    ROOT = os.getcwd()

    # ArcFace Loss
    atis_ArcFace, bank_ArcFace, clinc_ArcFace, hwu_ArcFace, mcid_ArcFace, rest_ArcFace = make_arcface(device)
    print('==================== ArcFace Losses Instantiated ====================')

    # Make DataLoader
    atis_tr_loader, atis_val_loader, atis_test_loader, bank_tr_loader, bank_val_loader, bank_test_loader ,\
    clinc_tr_loader, clinc_val_loader, clinc_test_loader, hwu_tr_loader, hwu_val_loader, hwu_test_loader, \
    mcid_tr_loader, mcid_val_loader, mcid_test_loader, rest_tr_loader, rest_val_loader, rest_test_loader = make_dataloader()
    print('==================== All DataLoader Loaded ====================')

    # Model
    model = MBD_Model(device=device, batch=Config.batch)
    model = model.to(device)
    tokenizer = Config.tokenizer
    print('==================== Model, Tokenizer Loaded ====================')

    train_steps = Config.epoch * (len(clinc_tr_loader))

    # Instantiate Optimizer, Loss function, Metric
    optimizer = AdamW(
                [{'params':model.parameters()},
                {'params': atis_ArcFace.parameters()},
                {'params': bank_ArcFace.parameters()},
                {'params': clinc_ArcFace.parameters()},
                {'params': hwu_ArcFace.parameters()},
                {'params': mcid_ArcFace.parameters()},
                {'params': rest_ArcFace.parameters()}],
                lr=Config.lr, 
                weight_decay=Config.weight_decay)

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=int(train_steps*0.1),
        num_training_steps=train_steps
        )

    ce_loss = nn.CrossEntropyLoss()

    train_metric = load_metric('accuracy')
    val_metric = load_metric('accuracy')
    test_metric = load_metric('accuracy')

    train_wrongs = []
    val_wrongs = []
    test_wrongs = []

    start_time = datetime.now()

    total_auc = 0

    auc_data = {'atis':[], 'bank':[],"clinc":[], "hwu":[], "mcid":[], "rest":[]}

    val_auc_dict = {'ATIS':[0.0], 'BANKING77':[0.0],"clinc150":[0.0], "HWU64":[0.0], "mcid":[0.0], "restaurant":[0.0]}
    # val_auc_dict = {'ATIS':[-1], 'BANKING77':[-1],"clinc150":[-1], "HWU64":[-1], "mcid":[-1], "restaurant":[-1]}
    epoch_list= [-1]
    # val_loss_dict = {'atis':[100], 'bank':[100],"clinc":[100], "hwu":[100], "mcid":[100], "rest":[100]}

    auc_data_test = {'atis':[], 'bank':[],"clinc":[], "hwu":[], "mcid":[], "rest":[]}

    total_info = {"epoch": [], "auc": [], "loss": []}
    best_AUC = 0.88
    best_loss = 0.3

    val_steps = (len(atis_val_loader)+len(bank_val_loader)+len(clinc_val_loader)+len(hwu_val_loader)+len(mcid_val_loader)+len(rest_val_loader)) * (train_steps//300)
    val_bar = tqdm(range(val_steps))

    for step in tqdm(range(1, train_steps + 1), total=train_steps, dynamic_ncols=True):

        if (step - 1) % len(atis_tr_loader) == 0:
            atis_tr_iter = iter(atis_tr_loader)
        if (step - 1) % len(bank_tr_loader) == 0:
            bank_tr_iter = iter(bank_tr_loader)
        if (step - 1) % len(clinc_tr_loader) == 0:
            clinc_tr_iter = iter(clinc_tr_loader)
        if (step - 1) % len(hwu_tr_loader) == 0:
            hwu_tr_iter = iter(hwu_tr_loader)
        if (step - 1) % len(mcid_tr_loader) == 0:
            mcid_tr_iter = iter(mcid_tr_loader)
        if (step - 1) % len(rest_tr_loader) == 0:
            rest_tr_iter = iter(rest_tr_loader)
        
        loss, auc, auc_data = train(model, tokenizer, atis_tr_iter, bank_tr_iter, clinc_tr_iter, hwu_tr_iter, mcid_tr_iter, rest_tr_iter,
                                        optimizer, lr_scheduler, ce_loss, train_metric, auc_data, train_wrongs, step, device)
        
        total_auc += auc

        # Validate Model for every 300 iterations
        interval = 300
        if step % interval == 0:
            # Validation 
            if step / interval == 1:
                print("Validation Bar")
                val_steps = (len(atis_val_loader)+len(bank_val_loader)+len(clinc_val_loader)+len(hwu_val_loader)+len(mcid_val_loader)+len(rest_val_loader)) * (train_steps//300)
                val_bar = tqdm(range(val_steps))
            atis_loss, atis_auc = validation(model, tokenizer,'atis', atis_val_loader, atis_ArcFace, optimizer, lr_scheduler, ce_loss, val_metric, val_wrongs, val_bar, device)        
            bank_loss, bank_auc = validation(model, tokenizer,'bank', bank_val_loader, bank_ArcFace, optimizer, lr_scheduler, ce_loss, val_metric, val_wrongs, val_bar, device)
            clinc_loss, clinc_auc = validation(model, tokenizer, 'clinc',clinc_val_loader, clinc_ArcFace, optimizer, lr_scheduler, ce_loss, val_metric, val_wrongs, val_bar, device)
            hwu_loss, hwu_auc = validation(model, tokenizer, 'hwu',hwu_val_loader, hwu_ArcFace, optimizer, lr_scheduler, ce_loss, val_metric, val_wrongs, val_bar, device)
            mcid_loss, mcid_auc = validation(model, tokenizer, 'mcid', mcid_val_loader, mcid_ArcFace, optimizer, lr_scheduler, ce_loss, val_metric, val_wrongs, val_bar, device)
            rest_loss, rest_auc = validation(model, tokenizer, 'rest', rest_val_loader, rest_ArcFace, optimizer, lr_scheduler, ce_loss, val_metric, val_wrongs, val_bar, device)

            val_loss = (atis_loss + bank_loss + clinc_loss + hwu_loss + mcid_loss + rest_loss) / 6
            val_auc = (atis_auc + bank_auc + clinc_auc + hwu_auc + mcid_auc + rest_auc) / 6

            # Log Test Info
            total_info["epoch"].append(step//interval)
            total_info["auc"].append(val_auc)
            total_info["loss"].append(val_loss)

            # Save results by dataset
            epoch_list.append(step//interval)
            val_auc_dict['ATIS'].append(atis_auc)
            val_auc_dict['BANKING77'].append(bank_auc)
            val_auc_dict['clinc150'].append(clinc_auc)
            val_auc_dict['HWU64'].append(hwu_auc)
            val_auc_dict['mcid'].append(mcid_auc)
            val_auc_dict['restaurant'].append(rest_auc)

            for key in val_auc_dict:
                fo = open(PATH + f"{key}/" + "AUC" + ".txt", 'a')
                fo.write(f'{key}        epoch:{epoch_list[-1]}        {str(val_auc_dict[key][-1])}\n')
                fo.close()
            
            saved = False
            for key in val_auc_dict:
                if (val_auc_dict[key][-1] > max(val_auc_dict[key][:-1])) :
                    fo = open(auc_data_file, 'a')
                    fo.write(f'{key}        epoch:{epoch_list[-1]}        {str(val_auc_dict[key][-1])}\n')
                    fo.close()
                    if not saved:
                        torch.save(model.state_dict(), PATH + f"{key}/" + f"{step//interval}" + ".pkl")
                        saved = True
                
            if (total_info['auc'][-1] > best_AUC):
                best_AUC = total_info['auc'][-1]
                # torch.save(model.state_dict(), PATH + "/ckpt/auc/" + f"{step//interval}" + ".pkl")
                fo = open(auc_file, "a")
                fo.write(f'epoch:{total_info["epoch"][-1]}       {str(total_info["auc"][-1])}\n')
                fo.close()

            if total_info['loss'][-1] < best_loss:
                best_loss = total_info['loss'][-1]
                # torch.save(model.state_dict(), PATH + "/ckpt/loss/" + f"{step//interval}" + ".pkl")
                fo = open(loss_file, "a")
                fo.write(f'epoch:{total_info["epoch"][-1]}       {str(total_info["loss"][-1])}\n')
                fo.close()

            auc_data = {'atis':[], 'bank':[],"clinc":[], "hwu":[], "mcid":[], "rest":[]}
            total_auc = 0
            
    fo = open(auc_data_file, 'a')
    fo.write('\n\n=========================================================================\n\n\n')
    fo.close()
    for key in val_auc_dict:
        fo = open(auc_data_file, 'a')
        idx = val_auc_dict[key].index(max(val_auc_dict[key]))
        fo.write(f'{key}        epoch:{epoch_list[idx]}        {str(max(val_auc_dict[key]))}\n')
        fo.close()

    fo = open(auc_file, 'a')
    fo.write('\n\n=========================================================================\n\n\n')
    fo.close()
    with open(auc_file, "a") as fo:
        for key in (val_auc_dict):
            fo.write(f'{key}        {val_auc_dict[key]}\n')
        fo.close()


    torch.save(model.state_dict(), PATH + "final.pkl")
