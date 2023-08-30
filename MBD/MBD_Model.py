import sys
import torch
from torch import nn
from transformers import AutoModel
from transformers import logging

logging.set_verbosity_error()


class MBD_Model(nn.Module):
    def __init__(self, device, batch):
        super(MBD_Model, self).__init__()
        
        # Instantiate Bert Layer
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pre_trained_model = MBD_Model(device=device, batch=batch)
pre_trained_model.to(device=device)
pre_trained_model.load_state_dict(torch.load('<path_to_saved_model_pkl_file>', map_location='cuda'))
