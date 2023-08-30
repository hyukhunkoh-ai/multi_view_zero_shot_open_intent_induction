import torch
import numpy as np
from tqdm.auto import tqdm
from torch import Tensor


def validation(model, tokenizer, dataset_name, val_loader, ArcFace, optimizer, lr_scheduler, \
                ce_loss, val_metric, val_wrongs, val_bar, device):
    with torch.set_grad_enabled(False):
        model.eval()
        
        total_auc = 0

        for utters, onehot, idx in val_loader:
            onehot = onehot.to(device)
            
            inputs =  tokenizer(list(utters), padding='longest',  truncation=True, return_tensors="pt")
            
            for key in inputs:
                if isinstance(inputs[key], Tensor):
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

