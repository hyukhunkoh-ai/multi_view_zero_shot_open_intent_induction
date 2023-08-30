import numpy as np
import torch
import numpy as np

from torch import nn, Tensor
from torch.nn import functional as F
from transformers import AutoModel,AutoTokenizer
from tqdm.autonotebook import trange
from typing import List, Union
from allennlp.common import Registrable
from sentence_transformers import SentenceTransformer


class SentenceEmbeddingModel(Registrable):
    def encode(self, utterances: List[str]) -> np.ndarray:
        """
        Encode a list of utterances as an array of real-valued vectors.
        :param utterances: original utterances
        :return: output encoding
        """
        raise NotImplementedError


@SentenceEmbeddingModel.register('sentence_transformers_model')
class SentenceTransformersModel(SentenceEmbeddingModel):

    def __init__(self, combination: str) -> None:
        """
        Initialize SentenceTransformers model for a given path or model name.
        :param model_name_or_path: model name or path for SentenceTransformers sentence encoder
        """
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Main_Model(device).to(device)
        s,t,k = list(combination.strip())
        print(s, t, k)
        self.use_sbert,self.use_mtl, self.use_kmeans = False,False,False
        if s == 't':
            self.use_sbert = True
        if t == 't':
            self.use_mtl = True
        if k == 't':
            self.use_kmeans = True            

    def encode(self,  utterances: List[str]) -> np.ndarray:
        with torch.no_grad():
           ret = self.model.encode(utterances,self.use_sbert,self.use_mtl, self.use_kmeans)
        return ret


def batch_to_device(batch, target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        
        self.bert_layer = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        
        self.device = device

    def forward(self, inputs, test, data=None):
        
        outputs = self.bert_latyer(**inputs)
        
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


pmodel = Model(device='cuda')
pmodel.to(device='cuda')
pmodel.load_state_dict(torch.load('sitod/saved_models/sbert_cls_20.pkl', map_location='cuda'))
plain_mtl = pmodel.bert_layer
kmodel = Model(device='cuda')
kmodel.to(device='cuda')
# kmodel.load_state_dict(torch.load('sitod/saved_models/sbert_cls_20.pkl', map_location='cuda'))
# kmodel.load_state_dict(torch.load('sitod/saved_models/e_10_lr_5e_6_kmeans_model_4.pt', map_location='cuda'))
kmodel.load_state_dict(torch.load('sitod/saved_models/TTF_e_10_lr_5e_6_kmeans_model_5.pt', map_location='cuda'))
# kmodel.load_state_dict(torch.load('sitod/saved_models/e_10_lr_5e_6_spectral_model_1.pt', map_location='cuda'))
kmeans_mtl = kmodel.bert_layer


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
    

class Main_Model(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
        self.sbert = mysbert().to(device)
        self.kmeans_mtl = kmeans_mtl
        self.plain_mtl = plain_mtl
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        
    def forward(self, raw_utter_batch, use_sbert=True, use_mtl=True, use_kmeans=True):
        
        input = self.tokenizer(raw_utter_batch, padding=True, truncation=True, return_tensors='pt')
        input = batch_to_device(input, self.device)

        
        if use_sbert:
            sbert_feat = self.sbert.encoder(**input)
            sbert_embedding = self.mean_pooling(sbert_feat, input['attention_mask'])
            sbert_embedding = F.normalize(sbert_embedding, p=2, dim=1)
    
        if use_kmeans:
            kmeans_feat = self.kmeans_mtl(**input)
            kmeans_embedding = self.mean_pooling(kmeans_feat, input['attention_mask'])
            kmeans_embedding = F.normalize(kmeans_embedding, p=2, dim=1)
        
        if use_mtl:
            plain_feat = self.plain_mtl(**input)
            plain_embedding = self.mean_pooling(plain_feat, input['attention_mask'])
            plain_embedding = F.normalize(plain_embedding, p=2, dim=1)

        # scaler1 = MinMaxScaler()
        # scaler2 = MinMaxScaler()
        # scaler1.fit(feature_list[:,:768])
        # scaler2.fit(feature_list[:,768:])
        # feature_list[:,:768] = scaler1.transform(feature_list[:,:768])
        # feature_list[:,768:] = scaler2.transform(feature_list[:,768:])

        if (use_sbert == True) & (use_kmeans == True) & (use_mtl ==True): 
            sentence_embedding = torch.cat((sbert_embedding, kmeans_embedding, plain_embedding), dim=1)

        if (use_sbert == True) & (use_kmeans == True) & (use_mtl == False):
            sentence_embedding = torch.cat((sbert_embedding, kmeans_embedding), dim=1)

        if (use_sbert == True) & (use_kmeans == False) & (use_mtl == True):
            sentence_embedding = torch.cat((sbert_embedding, plain_embedding), dim=1)

        if (use_sbert == False) & (use_kmeans == True) & (use_mtl == True):
            sentence_embedding = torch.cat((kmeans_embedding, plain_embedding), dim=1)


        if (use_sbert == True) & (use_kmeans == False) & (use_mtl == False):
            sentence_embedding = sbert_embedding
        if (use_sbert == False) & (use_kmeans == True) & (use_mtl == False):
            sentence_embedding = kmeans_embedding
        if (use_sbert == False) & (use_kmeans == False) & (use_mtl == True):
            sentence_embedding = plain_embedding
        if (use_sbert == False) & (use_kmeans == False) & (use_mtl == False):
            sbert_feat = self.sbert.encoder(**input)
            sentence_embedding = self.original_pooling(sbert_feat, input['attention_mask'])

        return sentence_embedding

    def original_pooling(self,model_output):
        return model_output[1]
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float().to(self.device)
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def _text_length(self, text: Union[List[int], List[List[int]]]):
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

    def encode(self, utterances, use_sbert=True, use_mtl=True, use_kmeans=True):        
        feature_list = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in utterances])
        utterances_sorted = [utterances[idx] for idx in length_sorted_idx]

        print('==================Make DataLoader==================')
        batch_size = 32; show_progress_bar =True
        for start_index in trange(0, len(utterances), batch_size, desc="Batches", disable=not show_progress_bar):
            
            raw_utter_batch = utterances_sorted[start_index : start_index+batch_size]

            with torch.no_grad():
                sentence_embeddings = self.forward(raw_utter_batch, use_sbert, use_mtl,use_kmeans)
                feature_list.extend(sentence_embeddings.detach().cpu()) 
            
        feature_list = [feature_list[idx] for idx in np.argsort(length_sorted_idx)]
        feature_list = np.asarray([emb.numpy() for emb in feature_list])
            
        return feature_list

