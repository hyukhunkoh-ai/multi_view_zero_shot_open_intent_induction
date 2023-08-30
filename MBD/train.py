import torch
import numpy as np
from torch import Tensor

def train(model, tokenizer, atis_tr_iter, bank_tr_iter, clinc_tr_iter, hwu_tr_iter, mcid_tr_iter, rest_tr_iter,
    atis_ArcFace, bank_ArcFace, clinc_ArcFace, hwu_ArcFace, mcid_ArcFace, rest_ArcFace, 
    optimizer, lr_scheduler, ce_loss, train_metric, auc_data, train_wrongs, step, device):
    with torch.set_grad_enabled(True):
        model.train()
        
        a_utter, a_onehot, a_idx = next(atis_tr_iter)
        b_utter, b_onehot, b_idx = next(bank_tr_iter)
        c_utter, c_onehot, c_idx = next(clinc_tr_iter)
        h_utter, h_onehot, h_idx = next(hwu_tr_iter)
        m_utter, m_onehot, m_idx = next(mcid_tr_iter)
        r_utter, r_onehot, r_idx = next(rest_tr_iter)

        a_onehot = a_onehot.to(device)
        b_onehot = b_onehot.to(device)
        c_onehot = c_onehot.to(device)
        h_onehot = h_onehot.to(device)
        m_onehot = m_onehot.to(device)
        r_onehot = r_onehot.to(device)

        utters = list(a_utter + b_utter + c_utter + h_utter + m_utter + r_utter)

        inputs =  tokenizer(utters, padding='longest',  truncation=True, return_tensors="pt")
        
        for key in inputs:
            if isinstance(inputs[key], Tensor):
                inputs[key] = inputs[key].to(device)

        atis, bank, clinc, hwu, mcid, rest = model(inputs, False, )
        
        cos_atis = atis_ArcFace(atis, a_onehot)
        cos_bank = bank_ArcFace(bank, b_onehot)
        cos_clinc = clinc_ArcFace(clinc, c_onehot)
        cos_hwu = hwu_ArcFace(hwu, h_onehot)
        cos_mcid = mcid_ArcFace(mcid, m_onehot)
        cos_rest = rest_ArcFace(rest, r_onehot)

        optimizer.zero_grad()

        loss = ce_loss(atis, a_onehot)+ ce_loss(bank, b_onehot) + ce_loss(clinc, c_onehot) *(10/14) +\
            ce_loss(hwu, h_onehot) + ce_loss(mcid, m_onehot) + ce_loss(rest, r_onehot)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        atis_pred = np.argmax(atis.cpu().detach().numpy(), axis=-1)
        bank_pred = np.argmax(bank.cpu().detach().numpy(), axis=-1)
        clinc_pred = np.argmax(clinc.cpu().detach().numpy(), axis=-1)
        hwu_pred = np.argmax(hwu.cpu().detach().numpy(), axis=-1)
        mcid_pred = np.argmax(mcid.cpu().detach().numpy(), axis=-1)
        rest_pred = np.argmax(rest.cpu().detach().numpy(), axis=-1)

        label_idx = np.concatenate((a_idx, b_idx, c_idx, h_idx, m_idx, r_idx), 0)
        preds = np.concatenate((atis_pred, bank_pred, clinc_pred, hwu_pred, mcid_pred, rest_pred), 0)

        total_auc = ((preds == label_idx).sum() / len(label_idx))
        
        auc_data['atis'].append((atis_pred == a_idx.cpu().detach().numpy()).sum() / len(a_idx))
        auc_data['bank'].append((bank_pred == b_idx.cpu().detach().numpy()).sum() / len(b_idx))
        auc_data["clinc"].append((clinc_pred == c_idx.cpu().detach().numpy()).sum() / len(c_idx))
        auc_data["hwu"].append((hwu_pred == h_idx.cpu().detach().numpy()).sum() / len(h_idx))
        auc_data["mcid"].append((mcid_pred == m_idx.cpu().detach().numpy()).sum() / len(m_idx) )
        auc_data["rest"].append((rest_pred == r_idx.cpu().detach().numpy()).sum() / len(r_idx))

        train_wrongs.extend([*label_idx[preds != label_idx]])
        
    return (loss/6), total_auc, auc_data


# def sort_by_length(utterances):


#     length_sorted_idx = np.argsort([-self._text_length(sen) for sen in utterances])
#     utterances_sorted = [utterances[idx] for idx in length_sorted_idx]


#     def _text_length(text: Union[List[int], List[List[int]]]):
#             """
#             Help function to get the length for the input text. Text can be either
#             a list of ints (which means a single text as input), or a tuple of list of ints
#             (representing several text inputs to the model).
#             """
#             if isinstance(text, dict):              #{key: value} case
#                 return len(next(iter(text.values())))
#             elif not hasattr(text, '__len__'):      #Object has no len() method
#                 return 1
#             elif len(text) == 0 or isinstance(text[0], int):    #Empty string or list of ints
#                 return len(text)
#             else:
#                 return sum([len(t) for t in text])      #Sum of length of individual strings