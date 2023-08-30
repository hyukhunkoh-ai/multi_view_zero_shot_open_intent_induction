
from torch import nn
from transformers import AutoModel
from dataset import *
from config import Config, ArcFace
from torch.utils.data import DataLoader

import torch
from torch import nn
from collections import defaultdict
from config import Config
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm.autonotebook import trange
from transformers import AutoModel


def make_arcface(device):
    atis_ArcFace = ArcFace(batch_size=Config.batch['ATIS'], in_feature=768, out_feature=Config.num_labels['ATIS'], s=Config.scale, m=Config.margin).to(device)
    bank_ArcFace = ArcFace(batch_size=Config.batch['BANKING77'], in_feature=768, out_feature=Config.num_labels['BANKING77'], s=Config.scale, m=Config.margin).to(device)
    clinc_ArcFace = ArcFace(batch_size=Config.batch['clinc150'], in_feature=768, out_feature=Config.num_labels['clinc150'], s=Config.scale, m=Config.margin).to(device)
    hwu_ArcFace = ArcFace(batch_size=Config.batch['HWU64'], in_feature=768, out_feature=Config.num_labels['HWU64'], s=Config.scale, m=Config.margin).to(device)
    mcid_ArcFace = ArcFace(batch_size=Config.batch['mcid'], in_feature=768, out_feature=Config.num_labels['mcid'], s=Config.scale, m=Config.margin).to(device)
    rest_ArcFace = ArcFace(batch_size=Config.batch['restaurant'], in_feature=768, out_feature=Config.num_labels['restaurant'], s=Config.scale, m=Config.margin).to(device)
    return atis_ArcFace, bank_ArcFace, clinc_ArcFace, hwu_ArcFace, mcid_ArcFace, rest_ArcFace




def make_dataloader():
    atis_tr, atis_val, atis_test = preprocessing('ATIS', val_frac=0.2, test_frac=0.1)
    bank_tr, bank_val, bank_test = preprocessing('BANKING77', val_frac=0.2, test_frac=0.1)
    clinc_tr, clinc_val, clinc_test = preprocessing('clinc150', val_frac=0.2, test_frac=0.1)
    hwu_tr, hwu_val, hwu_test = preprocessing('HWU64', val_frac=0.2, test_frac=0.1)
    mcid_tr, mcid_val, mcid_test = preprocessing('mcid', val_frac=0.2, test_frac=0.1)
    rest_tr, rest_val, rest_test = preprocessing('restaurant', val_frac=0.2, test_frac=0.1)
    
    print('==================== All Data Preprocessed ====================')

    atis_tr_loader = DataLoader(dataset=CLS_Dataset(df=atis_tr, num_labels=Config.num_labels['ATIS']), batch_size=Config.batch['ATIS'], shuffle=True, drop_last=True)
    atis_val_loader = DataLoader(dataset=CLS_Dataset(df=atis_val, num_labels=Config.num_labels['ATIS']), batch_size=Config.batch['ATIS'], shuffle=True, drop_last=True)
    atis_test_loader = DataLoader(dataset=CLS_Dataset(df=atis_test, num_labels=Config.num_labels['ATIS']), batch_size=Config.batch['ATIS'], shuffle=True, drop_last=True)

    bank_tr_loader = DataLoader(dataset=CLS_Dataset(df=bank_tr, num_labels=Config.num_labels['BANKING77']), batch_size=Config.batch['BANKING77'], shuffle=True, drop_last=True)
    bank_val_loader = DataLoader(dataset=CLS_Dataset(df=bank_val, num_labels=Config.num_labels['BANKING77']), batch_size=Config.batch['BANKING77'], shuffle=True, drop_last=True)
    bank_test_loader = DataLoader(dataset=CLS_Dataset(df=bank_test, num_labels=Config.num_labels['BANKING77']), batch_size=Config.batch['BANKING77'], shuffle=True, drop_last=True)

    clinc_tr_loader = DataLoader(dataset=CLS_Dataset(df=clinc_tr, num_labels=Config.num_labels['clinc150']), batch_size=Config.batch['clinc150'], shuffle=True, drop_last=True)
    clinc_val_loader = DataLoader(dataset=CLS_Dataset(df=clinc_val, num_labels=Config.num_labels['clinc150']), batch_size=Config.batch['clinc150'], shuffle=True, drop_last=True)
    clinc_test_loader = DataLoader(dataset=CLS_Dataset(df=clinc_test, num_labels=Config.num_labels['clinc150']), batch_size=Config.batch['clinc150'], shuffle=True, drop_last=True)

    hwu_tr_loader = DataLoader(dataset=CLS_Dataset(df=hwu_tr, num_labels=Config.num_labels['HWU64']), batch_size=Config.batch['HWU64'], shuffle=True, drop_last=True)
    hwu_val_loader = DataLoader(dataset=CLS_Dataset(df=hwu_val, num_labels=Config.num_labels['HWU64']), batch_size=Config.batch['HWU64'], shuffle=True, drop_last=True)
    hwu_test_loader = DataLoader(dataset=CLS_Dataset(df=hwu_test, num_labels=Config.num_labels['HWU64']), batch_size=Config.batch['HWU64'], shuffle=True, drop_last=True)

    mcid_tr_loader = DataLoader(dataset=CLS_Dataset(df=mcid_tr, num_labels=Config.num_labels['mcid']), batch_size=Config.batch['mcid'], shuffle=True, drop_last=True)
    mcid_val_loader = DataLoader(dataset=CLS_Dataset(df=mcid_val, num_labels=Config.num_labels['mcid']), batch_size=Config.batch['mcid'], shuffle=True, drop_last=True)
    mcid_test_loader = DataLoader(dataset=CLS_Dataset(df=mcid_test, num_labels=Config.num_labels['mcid']), batch_size=Config.batch['mcid'], shuffle=True, drop_last=True)

    rest_tr_loader = DataLoader(dataset=CLS_Dataset(df=rest_tr, num_labels=Config.num_labels['restaurant']), batch_size=Config.batch['restaurant'], shuffle=True, drop_last=True)
    rest_val_loader = DataLoader(dataset=CLS_Dataset(df=rest_val, num_labels=Config.num_labels['restaurant']), batch_size=Config.batch['restaurant'], shuffle=True, drop_last=True)
    rest_test_loader = DataLoader(dataset=CLS_Dataset(df=rest_test, num_labels=Config.num_labels['restaurant']), batch_size=Config.batch['restaurant'], shuffle=True, drop_last=True)
    
    return atis_tr_loader, atis_val_loader, atis_test_loader, bank_tr_loader, bank_val_loader, bank_test_loader ,\
        clinc_tr_loader, clinc_val_loader, clinc_test_loader, hwu_tr_loader, hwu_val_loader, hwu_test_loader, \
            mcid_tr_loader, mcid_val_loader, mcid_test_loader, rest_tr_loader, rest_val_loader, rest_test_loader



def load_pretrained_arcface(device):
    atis_ArcFace = ArcFace(batch_size=Config.batch['ATIS'], in_feature=768, out_feature=Config.num_labels['ATIS'], s=Config.scale, m=Config.margin).to(device)
    atis_ArcFace.load_state_dict(torch.load('./arcfaces/ATIS/2.pkl', map_location='cuda'))
    
    bank_ArcFace = ArcFace(batch_size=Config.batch['BANKING77'], in_feature=768, out_feature=Config.num_labels['BANKING77'], s=Config.scale, m=Config.margin).to(device)
    bank_ArcFace.load_state_dict(torch.load('./arcfaces/BANKING77/2.pkl', map_location='cuda'))
    
    clinc_ArcFace = ArcFace(batch_size=Config.batch['clinc150'], in_feature=768, out_feature=Config.num_labels['clinc150'], s=Config.scale, m=Config.margin).to(device)
    clinc_ArcFace.load_state_dict(torch.load('./arcfaces/clinc150/30.pkl', map_location='cuda'))
    
    hwu_ArcFace = ArcFace(batch_size=Config.batch['HWU64'], in_feature=768, out_feature=Config.num_labels['HWU64'], s=Config.scale, m=Config.margin).to(device)
    hwu_ArcFace.load_state_dict(torch.load('./arcfaces/HWU64/27.pkl', map_location='cuda'))
    
    mcid_ArcFace = ArcFace(batch_size=Config.batch['mcid'], in_feature=768, out_feature=Config.num_labels['mcid'], s=Config.scale, m=Config.margin).to(device)
    mcid_ArcFace.load_state_dict(torch.load('./arcfaces/mcid/2.pkl', map_location='cuda'))
    
    rest_ArcFace = ArcFace(batch_size=Config.batch['restaurant'], in_feature=768, out_feature=Config.num_labels['restaurant'], s=Config.scale, m=Config.margin).to(device)
    rest_ArcFace.load_state_dict(torch.load('./arcfaces/restaurant/1.pkl', map_location='cuda'))
    
    return atis_ArcFace, bank_ArcFace, clinc_ArcFace, hwu_ArcFace, mcid_ArcFace, rest_ArcFace

class MBD(nn.Module):
    def __init__(self, device, batch):
        super(MBD, self).__init__()
        
        # Instantiate Bert Layer
        # self.bert_layer = AlbertModel.from_pretrained("albert-base-v2")
        self.bert_layer = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        print("==================== Bert Layer Loaded ====================")
        
        self.device = device
   
        self.batch_a = batch['ATIS']
        self.batch_b = self.batch_a + batch['BANKING77']
        self.batch_c = self.batch_b + batch['clinc150']
        self.batch_h = self.batch_c + batch['HWU64']
        self.batch_m = self.batch_h + batch['mcid']

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs, test, data=None):
        
        outputs = self.bert_layer(**inputs)
        
        pooled_output = outputs[1] 
        pooled_output = self.dropout(pooled_output)

        if not test:   
            atis = pooled_output[:self.batch_a,:]
            bank = pooled_output[self.batch_a:self.batch_b,:]
            clinc = pooled_output[self.batch_b:self.batch_c,:]
            hwu = pooled_output[self.batch_c:self.batch_h,:]
            mcid = pooled_output[self.batch_h:self.batch_m,:]
            rest = pooled_output[self.batch_m:,:]

            return atis.to(self.device), bank.to(self.device), clinc.to(self.device),\
                hwu.to(self.device), mcid.to(self.device), rest.to(self.device)
        else:
            
            return pooled_output

batch = {"ATIS": 10, "BANKING77" :10, "clinc150":14, "HWU64":10, "mcid":10, "restaurant":10}

pre_trained_model = MBD(device='cuda', batch=batch)
pre_trained_model.to(device='cuda')
pre_trained_model.load_state_dict(torch.load('./sbert_cls_20.pkl', map_location='cuda'),strict=False)


class ClusterModel(nn.Module):
    def __init__(self, device, batch):
        super(ClusterModel, self).__init__()
        
        # Instantiate Bert Layer
        self.device = device
        self.mtl_bert = pre_trained_model.bert_layer.to(device)
        self.sbert = mysbert().to(device)
        print("==================== Bert Layer Loaded ====================")

        self.batch_a = batch['ATIS']
        self.batch_b = self.batch_a + batch['BANKING77']
        self.batch_c = self.batch_b + batch['clinc150']
        self.batch_h = self.batch_c + batch['HWU64']
        self.batch_m = self.batch_h + batch['mcid']

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(Config.dropout_prob)

        self.encoded_inputs = {}

    def forward(self, features, test, data=None):

        temp_features = self.sbert.encoder(**features)    
        mtl_features = self.mtl_bert(**features)
        sentence_embeddings0 = mean_pooling(temp_features, features['attention_mask'])
        sentence_embeddings0 = F.normalize(sentence_embeddings0, p=2, dim=1)

        sentence_embeddings1 = mtl_mean_pooling(mtl_features, features['attention_mask'])
        sentence_embeddings1 = F.normalize(sentence_embeddings1, p=2, dim=1)
        pooled_output = torch.cat((sentence_embeddings0,sentence_embeddings1),dim=1)
        pooled_output = self.dropout(pooled_output)
        if not test:   
            atis = pooled_output[:self.batch_a,:]
            bank = pooled_output[self.batch_a:self.batch_b,:]
            clinc = pooled_output[self.batch_b:self.batch_c,:]
            hwu = pooled_output[self.batch_c:self.batch_h,:]
            mcid = pooled_output[self.batch_h:self.batch_m,:]
            rest = pooled_output[self.batch_m:,:]
            
            return atis.to(self.device), bank.to(self.device), clinc.to(self.device),\
                hwu.to(self.device), mcid.to(self.device), rest.to(self.device)
        else:
            return pooled_output

    def assign_cluster(self, clustering_algorithm, train_features, utterance_by_id, intent_by_id, turn_ids):

        clusters = clustering_algorithm.fit(train_features)
        cluster_assignments = {turn_id: str(label) for turn_id, label in zip(turn_ids, clusters)}
        
        print('=======================clustering process finished=======================')
        utterances_by_cluster_id = defaultdict(list)
        for turn_id, cluster_label in cluster_assignments.items():
            utterances_by_cluster_id[cluster_label].append(utterance_by_id[turn_id])
            utterances_by_cluster_id[cluster_label].append(intent_by_id[turn_id])
        return utterances_by_cluster_id, cluster_assignments

    def alignment(self, km, args):

        if self.centroids is not None:

            old_centroids = self.centroids.cpu().numpy()
            new_centroids = km.cluster_centers_
            
            DistanceMatrix = np.linalg.norm(old_centroids[:,np.newaxis,:]-new_centroids[np.newaxis,:,:],axis=2) 
            row_ind, col_ind = linear_sum_assignment(DistanceMatrix)
            
            new_centroids = torch.tensor(new_centroids).to(self.device)
            self.centroids = torch.empty(self.num_labels ,args.feat_dim).to(self.device)
            
            alignment_labels = list(col_ind)
            for i in range(self.num_labels):
                label = alignment_labels[i]
                self.centroids[i] = new_centroids[label]
                
            pseudo2label = {label:i for i,label in enumerate(alignment_labels)}
            pseudo_labels = np.array([pseudo2label[label] for label in km.labels_])

        else:
            self.centroids = torch.tensor(km.cluster_centers_).to(self.device)        
            pseudo_labels = km.labels_ 

        pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long).to(self.device)
        
        return pseudo_labels

    def _text_length(self, text):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """
        if isinstance(text, dict):              #{key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):      #Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):    #Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])      #Sum of length of individual strings

    def encode(self, utterances, device, use_sbert=True, use_mtl=True, use_srl=False):
        print('==================Encoding==================')
        
        feature_list = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in utterances])
        utterances_sorted = [utterances[idx] for idx in length_sorted_idx]

        print('==================Make DataLoader==================')
        batch_size = 64; show_progress_bar =True
        for start_index in trange(0, len(utterances), batch_size, desc="Batches", disable=not show_progress_bar):
            
            raw_utter_batch = utterances_sorted[start_index : start_index+batch_size]

            with torch.no_grad():
                sentence_embeddings = self.forward(raw_utter_batch, use_sbert, use_mtl, use_srl)
                feature_list.extend(sentence_embeddings.detach().cpu()) 
            
        feature_list = [feature_list[idx] for idx in np.argsort(length_sorted_idx)]
        feature_list = np.asarray([emb.numpy() for emb in feature_list])
            
        return feature_list



class mysbert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        

        self.encoder = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')


        self.vocab_size = 30527
        self.hidden_size = 768
        bias = torch.nn.Parameter(torch.zeros(self.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.projection_lm = torch.nn.Linear(self.hidden_size,self.vocab_size)
        self.projection_lm.weight = self.encoder.embeddings.word_embeddings.weight
        self.projection_lm.bias = bias
    
    def mask_forward(self, masked_utters,input_masks):
        # (bs, n_enc_seq, d_hidn), (bs, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        sequence_output = self.encoder(input_ids = masked_utters,attention_mask=input_masks)
        # print(sequence_output['last_hidden_state'].shape) #torch.Size([64, 107, 768])
        # print(sequence_output['pooler_output'].shape) #torch.Size([64, 768])
        

        # (bs, n_enc_seq, n_enc_vocab)
        logits_lm = self.projection_lm(sequence_output['last_hidden_state'])
        
        return logits_lm
    def contrastive_forward(self, utters,masked_utters,input_masks):
        # bs,seq + bs,seq 
        cl_utters = torch.cat((utters,masked_utters),1).view((-1,utters.size(-1)))
        cl_input_masks = torch.cat((input_masks,input_masks),1).view((-1,utters.size(-1)))
        sequence_output = self.encoder(input_ids = cl_utters,attention_mask=cl_input_masks)
        output = sequence_output['last_hidden_state']
        pair_output = output.view(-1,2,output.size(-1))
        z1 = pair_output[:,0]
        z2 = pair_output[:,1]

        return z1,z2

    def calculate_cos_sim(self,x,y,temp=0.1):
        self.temp = temp
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        return self.cos(x, y) / self.temp
        

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    # attention_mask = torch.ones(token_embeddings.size(0),token_embeddings.size(1)).to("cuda")
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    # return token_embeddings

def mtl_mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    # attention_mask = torch.ones(token_embeddings.size(0),token_embeddings.size(1)).to("cuda")
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    # return token_embeddings

class PGT_Model(nn.Module):
    def __init__(self, device, batch):
        super(PGT_Model, self).__init__()
        
        # Instantiate Bert Layer
        self.device = device
        self.bert_layer = pre_trained_model.bert_layer.to(device)
        print("==================== Bert Layer Loaded ====================")

        self.batch_a = batch['ATIS']
        self.batch_b = self.batch_a + batch['BANKING77']
        self.batch_c = self.batch_b + batch['clinc150']
        self.batch_h = self.batch_c + batch['HWU64']
        self.batch_m = self.batch_h + batch['mcid']

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(Config.dropout_prob)

        self.encoded_inputs = {}

    def forward(self, inputs, test, data=None):
        
        outputs = self.bert_layer(**inputs)
        
        pooled_output = outputs[1] 
        pooled_output = self.dropout(pooled_output)

        if not test:   
            atis = pooled_output[:self.batch_a,:]
            bank = pooled_output[self.batch_a:self.batch_b,:]
            clinc = pooled_output[self.batch_b:self.batch_c,:]
            hwu = pooled_output[self.batch_c:self.batch_h,:]
            mcid = pooled_output[self.batch_h:self.batch_m,:]
            rest = pooled_output[self.batch_m:,:]
            
            return atis.to(self.device), bank.to(self.device), clinc.to(self.device),\
                hwu.to(self.device), mcid.to(self.device), rest.to(self.device)
        else:
            return outputs  # pooled_output

    def assign_cluster(self, clustering_algorithm, train_features, utterance_by_id, intent_by_id, turn_ids):

        clusters = clustering_algorithm.fit(train_features)
        cluster_assignments = {turn_id: str(label) for turn_id, label in zip(turn_ids, clusters)}
        
        print('=======================clustering process finished=======================')
        utterances_by_cluster_id = defaultdict(list)
        for turn_id, cluster_label in cluster_assignments.items():
            utterances_by_cluster_id[cluster_label].append(utterance_by_id[turn_id])
            utterances_by_cluster_id[cluster_label].append(intent_by_id[turn_id])
        return utterances_by_cluster_id, cluster_assignments

    def alignment(self, km, args):

        if self.centroids is not None:

            old_centroids = self.centroids.cpu().numpy()
            new_centroids = km.cluster_centers_
            
            DistanceMatrix = np.linalg.norm(old_centroids[:,np.newaxis,:]-new_centroids[np.newaxis,:,:],axis=2) 
            row_ind, col_ind = linear_sum_assignment(DistanceMatrix)
            
            new_centroids = torch.tensor(new_centroids).to(self.device)
            self.centroids = torch.empty(self.num_labels ,args.feat_dim).to(self.device)
            
            alignment_labels = list(col_ind)
            for i in range(self.num_labels):
                label = alignment_labels[i]
                self.centroids[i] = new_centroids[label]
                
            pseudo2label = {label:i for i,label in enumerate(alignment_labels)}
            pseudo_labels = np.array([pseudo2label[label] for label in km.labels_])

        else:
            self.centroids = torch.tensor(km.cluster_centers_).to(self.device)        
            pseudo_labels = km.labels_ 

        pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long).to(self.device)
        
        return pseudo_labels

    def _text_length(self, text):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """
        if isinstance(text, dict):              #{key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):      #Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):    #Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])      #Sum of length of individual strings

    def encode(self, utterances, device, use_sbert=True, use_mtl=True, use_srl=False):
        print('==================Encoding==================')
        
        feature_list = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in utterances])
        utterances_sorted = [utterances[idx] for idx in length_sorted_idx]

        print('==================Make DataLoader==================')
        batch_size = 64; show_progress_bar =True
        for start_index in trange(0, len(utterances), batch_size, desc="Batches", disable=not show_progress_bar):
            
            raw_utter_batch = utterances_sorted[start_index : start_index+batch_size]

            with torch.no_grad():
                sentence_embeddings = self.forward(raw_utter_batch, use_sbert, use_mtl, use_srl)
                feature_list.extend(sentence_embeddings.detach().cpu()) 
            
        feature_list = [feature_list[idx] for idx in np.argsort(length_sorted_idx)]
        feature_list = np.asarray([emb.numpy() for emb in feature_list])
            
        return feature_list


def validation(model, tokenizer, dataset_name, val_loader, ArcFace, optimizer, lr_scheduler, \
                ce_loss, val_metric, val_wrongs, val_bar, device):
    with torch.set_grad_enabled(False):
        model.eval()
        
        total_auc = 0

        for utters, onehot, idx in val_loader:
            onehot = onehot.to(device)
            
            inputs =  tokenizer(list(utters), padding='longest',  truncation=True, return_tensors="pt")
            
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(device)

            output = model(inputs, True, dataset_name)
            
            cos_phi = ArcFace(output, onehot)
            loss = ce_loss(cos_phi, onehot)
            
            pred = np.argmax(cos_phi.cpu().detach().numpy(), axis=-1)
            label_idx = idx.cpu().detach().numpy()

            total_auc += ((pred==label_idx).sum()) / len(label_idx)

            val_wrongs.extend([*label_idx[pred != label_idx]])
            val_bar.update(1)
            
    return (loss),  total_auc / len(val_loader)

