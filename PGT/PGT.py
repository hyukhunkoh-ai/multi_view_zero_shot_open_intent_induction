import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import scipy
import gc

from tqdm.auto import tqdm
from config import Config
from dataset import Teacher_Dataset
from trainer import PGT_Model, ClusterModel, load_pretrained_arcface, validation # validation might be necessary
from sklearn.cluster import KMeans

from torch import nn
from torch.optim import AdamW

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

gc.collect()
torch.cuda.empty_cache()    

# Make Directories 

data_list = ['ATIS', 'BANKING77', 'clinc150','HWU64','mcid', 'restaurant']

print("==================== results Created ====================")


# Print Device Info
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:{device}')

if torch.cuda.is_available():
    print("Current cuda device: ", torch.cuda.current_device())
    print("Count of Using GPUs: ", torch.cuda.device_count())

# Make Teacher DataLoader
nlabels = []
hidden_dim = 768
cluster_hidden_dim = 1536

tokenizer = Config.tokenizer
atis_teacher =Teacher_Dataset(path ='./data/ATIS.csv')
bank_teacher =Teacher_Dataset(path = './data/BANKING77.csv')
clinc_teacher =Teacher_Dataset(path = './data/clinc150.csv')
hwu_teacher =Teacher_Dataset(path = './data/HWU64.csv')
mcid_teacher =Teacher_Dataset(path = './data/mcid.csv')
rest_teacher =Teacher_Dataset(path = './data/restaurant.csv')

nlabels.append(atis_teacher.num_labels)
nlabels.append(bank_teacher.num_labels)
nlabels.append(clinc_teacher.num_labels)
nlabels.append(hwu_teacher.num_labels)
nlabels.append(mcid_teacher.num_labels)
nlabels.append(rest_teacher.num_labels)

atis_kmeans = torch.utils.data.DataLoader(dataset=atis_teacher, batch_size=256,shuffle=False)
bank_kmeans = torch.utils.data.DataLoader(dataset=bank_teacher, batch_size=256,shuffle=False)
clinc_kmeans = torch.utils.data.DataLoader(dataset=clinc_teacher, batch_size=256,shuffle=False)
hwu_kmeans = torch.utils.data.DataLoader(dataset=hwu_teacher, batch_size=256,shuffle=False)
mcid_kmeans = torch.utils.data.DataLoader(dataset=mcid_teacher, batch_size=256,shuffle=False)
rest_kmeans = torch.utils.data.DataLoader(dataset=rest_teacher, batch_size=256,shuffle=False)
# output : index, utter, label

total_dataset =[atis_teacher,bank_teacher,clinc_teacher,hwu_teacher,mcid_teacher,rest_teacher]
total_loader = [atis_kmeans, bank_kmeans,clinc_kmeans,hwu_kmeans,mcid_kmeans,rest_kmeans]
cluster_model = ClusterModel(device=device, batch=Config.batch)
kmeans_model = PGT_Model(device=device, batch=Config.batch)
teacher_model = PGT_Model(device=device, batch=Config.batch)

kmeans_model.train()
teacher_model.train()
# for param in teacher_model.parameters():
#     param.requires_grad = True

optimizer = AdamW(
            [{'params':kmeans_model.parameters()}],
            lr=Config.lr, 
            weight_decay=Config.weight_decay)

atis_ArcFace, bank_ArcFace, clinc_ArcFace, hwu_ArcFace, mcid_ArcFace, rest_ArcFace = load_pretrained_arcface(device)
atis_ArcFace.eval(); bank_ArcFace.eval(); clinc_ArcFace.eval(); hwu_ArcFace.eval(); mcid_ArcFace.eval(); rest_ArcFace.eval()

train_steps = None
old_centroids = {i:None for i in data_list}

best_loss = 10000
num_iter = 0

for epoch in tqdm(range(int(Config.epoch))):
    total_loss = 0
    dnames = {i:[] for i in data_list}
    #### encoding & pseudo label assignment ###
    with torch.no_grad():
        for trdata_loader,dname in tqdm(zip(total_loader,dnames.keys()), desc='datalodaer iter', total=len(total_loader)):
            for step,data in tqdm(enumerate(trdata_loader), desc='dataloader_encode', total=len(trdata_loader)):
                # print(len(data))
                
                features = tokenizer(list(data[1]), padding=True, truncation=True, return_tensors='pt')
                features['input_ids'] = features['input_ids'].to(device)
                features['attention_mask'] = features['attention_mask'].to(device)

                mtl_features = cluster_model(features,True)
                dnames[dname].extend(mtl_features.detach().cpu().numpy())       

    # cluster with alignment
    for idx, (features, num) in tqdm(enumerate(zip(dnames.values(), nlabels)), desc='kmeans iter by dataset', total=len(nlabels)):
        km = KMeans(n_clusters=num, n_init=10)
        km.fit(np.asarray(features))
        
        if old_centroids[data_list[idx]] is not None:
            # New Centroids to update
            new_centroids = km.cluster_centers_
            # For Align new label
            DistanceMatrix = np.linalg.norm(old_centroids[data_list[idx]][:,np.newaxis,:]-new_centroids[np.newaxis,:,:], axis=2)
            _, col_ind = scipy.optimize.linear_sum_assignment(DistanceMatrix)
            old_centroids[data_list[idx]] = torch.empty(num,cluster_hidden_dim)
            alignment_labels = list(col_ind)



            for i in range(num):
                label = np.asarray(alignment_labels[i])
                old_centroids[data_list[idx]][i] = torch.from_numpy(new_centroids[label])
            pseudo2label = {label:i for i,label in enumerate(alignment_labels)}
            total_dataset[idx].df['pseudo_label']  = np.array([pseudo2label[label] for label in km.labels_])
        else:
            old_centroids[data_list[idx]] = km.cluster_centers_
            total_dataset[idx].df['pseudo_label'] = km.labels_

    ##### pseudo label assignment finished####

    ###### make new dataloader for training #####
    atis_tr_iter = iter(torch.utils.data.DataLoader(dataset=atis_teacher, batch_size=Config.batch['ATIS'],shuffle=True,drop_last=True))
    bank_tr_iter = iter(torch.utils.data.DataLoader(dataset=bank_teacher, batch_size=Config.batch['BANKING77'],shuffle=True,drop_last=True))
    clinc_tr_iter = iter(torch.utils.data.DataLoader(dataset=clinc_teacher, batch_size=Config.batch['clinc150'],shuffle=True,drop_last=True))
    hwu_tr_iter = iter(torch.utils.data.DataLoader(dataset=hwu_teacher, batch_size=Config.batch['HWU64'],shuffle=True,drop_last=True))
    mcid_tr_iter = iter(torch.utils.data.DataLoader(dataset=mcid_teacher, batch_size=Config.batch['mcid'],shuffle=True,drop_last=True))
    rest_tr_iter = iter(torch.utils.data.DataLoader(dataset=rest_teacher, batch_size=Config.batch['restaurant'],shuffle=True,drop_last=True))
    

    if train_steps is None:
        train_steps = Config.epoch * (len(clinc_tr_iter))

    #### batch iter loop ###
    with torch.set_grad_enabled(True):

        for step in tqdm(range(1, len(mcid_tr_iter))):
            aindex, autter, alabel, apseudo = next(atis_tr_iter)
            bindex, butter, blabel, bpseudo = next(bank_tr_iter)
            cindex, cutter, clabel, cpseudo = next(clinc_tr_iter)
            hindex, hutter, hlabel, hpseudo = next(hwu_tr_iter)
            mindex, mutter, mlabel, mpseudo = next(mcid_tr_iter)
            rindex, rutter, rlabel, rpseudo = next(rest_tr_iter)
            
            alabel, blabel, clabel, hlabel, mlabel, rlabel = alabel.to(device), blabel.to(device), clabel.to(device), hlabel.to(device), mlabel.to(device), rlabel.to(device)

            apseudo = apseudo.type(torch.LongTensor).to(device)
            bpseudo = bpseudo.type(torch.LongTensor).to(device)
            cpseudo = cpseudo.type(torch.LongTensor).to(device)
            hpseudo = hpseudo.type(torch.LongTensor).to(device)
            mpseudo = mpseudo.type(torch.LongTensor).to(device)
            rpseudo = rpseudo.type(torch.LongTensor).to(device)

            utters = list(autter + butter + cutter + hutter + mutter + rutter)

            inputs =  tokenizer(utters, padding=True,  truncation=True, return_tensors="pt")
                
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(device)
                
            optimizer.zero_grad()

            atis, bank, clinc, hwu, mcid, rest = teacher_model(inputs, False)

            a_onehot = F.one_hot(apseudo, num_classes=atis_teacher.num_labels).float().to(device)
            b_onehot = F.one_hot(bpseudo, num_classes=bank_teacher.num_labels).float().to(device)
            c_onehot = F.one_hot(cpseudo, num_classes=clinc_teacher.num_labels).float().to(device)
            h_onehot = F.one_hot(hpseudo, num_classes=hwu_teacher.num_labels).float().to(device)
            m_onehot = F.one_hot(mpseudo, num_classes=mcid_teacher.num_labels).float().to(device)
            r_onehot = F.one_hot(rpseudo, num_classes=rest_teacher.num_labels).float().to(device)

            cos_atis = atis_ArcFace(atis, a_onehot)
            cos_bank = bank_ArcFace(bank, b_onehot)
            cos_clinc = clinc_ArcFace(clinc, c_onehot)
            cos_hwu = hwu_ArcFace(hwu, h_onehot)
            cos_mcid = mcid_ArcFace(mcid, m_onehot)
            cos_rest = rest_ArcFace(rest, r_onehot)

            ce_loss = nn.CrossEntropyLoss()

            loss = ce_loss(cos_atis, apseudo)+ ce_loss(cos_bank, bpseudo) + ce_loss(cos_clinc, cpseudo) *(10/14) +\
                ce_loss(cos_hwu, hpseudo) + ce_loss(cos_mcid, mpseudo) + ce_loss(cos_rest, rpseudo)


            total_loss += loss
            loss.backward()

            num_iter += 1
            
            if step == 1:
                for kmeans,teacher in zip(kmeans_model.named_parameters(),teacher_model.named_parameters()):
                    if teacher[1].grad is not None:
                        kmeans[1].grad = teacher[1].grad.clone()
                    else:
                        pass
            else:
                for kmeans,teacher in zip(kmeans_model.named_parameters(),teacher_model.named_parameters()):
                    if teacher[1].grad is not None:
                        tmp = kmeans[1] - teacher[1]
                        kmeans[1].grad = teacher[1].grad.clone() + tmp
                    else:
                        pass

            optimizer.step()

            teacher_model.zero_grad()



    if (total_loss < best_loss):
        best_loss = total_loss
    
        if epoch == 0 :
            continue
        torch.save(kmeans_model.state_dict(), f"./TTF_e_10_lr_5e_6_kmeans_model_{epoch}.pt")