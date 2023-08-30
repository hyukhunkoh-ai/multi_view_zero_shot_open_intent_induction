from MBD.dataset import CLS_Dataset, preprocessing
from MBD.loss import ArcFace
from MBD.config import Config
from torch.utils.data import  DataLoader
import torch


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

# ArcFace Loss
def make_arcface(device):
    atis_ArcFace = ArcFace(batch_size=Config.batch['ATIS'], in_feature=768, out_feature=Config.num_labels['ATIS'], s=Config.scale, m=Config.margin).to(device)
    bank_ArcFace = ArcFace(batch_size=Config.batch['BANKING77'], in_feature=768, out_feature=Config.num_labels['BANKING77'], s=Config.scale, m=Config.margin).to(device)
    clinc_ArcFace = ArcFace(batch_size=Config.batch['clinc150'], in_feature=768, out_feature=Config.num_labels['clinc150'], s=Config.scale, m=Config.margin).to(device)
    hwu_ArcFace = ArcFace(batch_size=Config.batch['HWU64'], in_feature=768, out_feature=Config.num_labels['HWU64'], s=Config.scale, m=Config.margin).to(device)
    mcid_ArcFace = ArcFace(batch_size=Config.batch['mcid'], in_feature=768, out_feature=Config.num_labels['mcid'], s=Config.scale, m=Config.margin).to(device)
    rest_ArcFace = ArcFace(batch_size=Config.batch['restaurant'], in_feature=768, out_feature=Config.num_labels['restaurant'], s=Config.scale, m=Config.margin).to(device)
    return atis_ArcFace, bank_ArcFace, clinc_ArcFace, hwu_ArcFace, mcid_ArcFace, rest_ArcFace

def load_pretrained_arcface(device):
    atis_ArcFace = ArcFace(batch_size=Config.batch['ATIS'], in_feature=768, out_feature=Config.num_labels['ATIS'], s=Config.scale, m=Config.margin).to(device)
    atis_ArcFace.load_state_dict(torch.load('<path_to_Arcface_pkl_file_for_ATIS>', map_location='cuda'))
    
    bank_ArcFace = ArcFace(batch_size=Config.batch['BANKING77'], in_feature=768, out_feature=Config.num_labels['BANKING77'], s=Config.scale, m=Config.margin).to(device)
    bank_ArcFace.load_state_dict(torch.load('<path_to_Arcface_pkl_file_for_BANKING>', map_location='cuda'))
    
    clinc_ArcFace = ArcFace(batch_size=Config.batch['clinc150'], in_feature=768, out_feature=Config.num_labels['clinc150'], s=Config.scale, m=Config.margin).to(device)
    clinc_ArcFace.load_state_dict(torch.load('<path_to_Arcface_pkl_file_for_clinc150>', map_location='cuda'))
    
    hwu_ArcFace = ArcFace(batch_size=Config.batch['HWU64'], in_feature=768, out_feature=Config.num_labels['HWU64'], s=Config.scale, m=Config.margin).to(device)
    hwu_ArcFace.load_state_dict(torch.load('<path_to_Arcface_pkl_file_for_HWU64>', map_location='cuda'))
    
    mcid_ArcFace = ArcFace(batch_size=Config.batch['mcid'], in_feature=768, out_feature=Config.num_labels['mcid'], s=Config.scale, m=Config.margin).to(device)
    mcid_ArcFace.load_state_dict(torch.load('<path_to_Arcface_pkl_file_for_mcid>', map_location='cuda'))
    
    rest_ArcFace = ArcFace(batch_size=Config.batch['restaurant'], in_feature=768, out_feature=Config.num_labels['restaurant'], s=Config.scale, m=Config.margin).to(device)
    rest_ArcFace.load_state_dict(torch.load('<path_to_Arcface_pkl_file_for_restaurant>', map_location='cuda'))
    
    return atis_ArcFace, bank_ArcFace, clinc_ArcFace, hwu_ArcFace, mcid_ArcFace, rest_ArcFace

