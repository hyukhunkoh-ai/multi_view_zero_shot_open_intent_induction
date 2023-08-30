import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class Teacher_Dataset(Dataset):
    def __init__(self, path):
        super(Dataset, self).__init__()

        self.df = pd.read_csv(path, sep='|')

        labels = list(self.df['intent'].unique())
        if 'no_intent' in labels:
            self.df = self.df.loc[self.df.loc[:,'intent'] != 'no_intent'].reset_index().drop(columns='index')
            labels = list(self.df['intent'].unique())
        
        self.df = self.df.sort_values(by='utterance',key=lambda col:np.argsort([-len(sen) for sen in col]))
        self.df['pseudo_label'] = np.ones(len(self.df))
        self.num_labels = len(labels)
        self.df['label_id'] = pd.factorize(self.df['intent'])[0]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index):

        utter = self.df['utterance'][index]
        label = self.df['label_id'][index]
        pseudo = self.df['pseudo_label'][index]

        return index, utter, label, pseudo


def preprocessing(dataset_name :str,  val_frac, test_frac):

    df = pd.read_csv(f'<path_to_dataset>', sep="|")
    # sort by length
    df = df.sort_values(by="utterance", key=lambda x: x.str.len()).reset_index(drop=True)
    print(f"==================== {dataset_name} loaded ====================")
    
    labels = list(df['intent'].unique())
    if 'no_intent' in labels:
        df = df.loc[df.loc[:,'intent'] != 'no_intent'].reset_index().drop(columns='index')
        labels = list(df['intent'].unique())

    df['label_idx'] = pd.factorize(df['intent'])[0]
    
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    test_frac /= (1-val_frac)

    for la in labels:
        tmp = df.loc[df['intent'] == la].sample(frac=val_frac, replace=False)
        val_df = pd.concat((val_df, tmp), axis=0)
        
        df = pd.concat((df, tmp), axis=0).drop_duplicates(keep=False)

        tmp = df.loc[df['intent'] == la].sample(frac=test_frac, replace=False)
        test_df = pd.concat((test_df, tmp), axis=0)
        df = pd.concat((df, tmp), axis=0).drop_duplicates(keep=False)
    
    df = df.reset_index().drop(columns='index')
    val_df = val_df.reset_index().drop(columns='index')
    test_df = test_df.reset_index().drop(columns='index')

    return df, val_df, test_df


class CLS_Dataset(Dataset):
    def __init__(self, df, num_labels):
        super(Dataset, self).__init__()

        self.df = df
        self.num_labels = num_labels
        label_idx = df.loc[:, 'label_idx']
        label_idx = torch.tensor(label_idx)
        label_onehot = F.one_hot(label_idx, num_classes=num_labels)
        label_onehot = label_onehot.float()
        df['label_onehot'] = label_onehot.tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index):
        
        # tokenized_utter =  self.tokenizer(self.df.loc[index, 'utterance'], padding='longest',  truncation=True, return_tensors="pt")
        
        label_onehot = self.df.loc[index, 'label_onehot']
        label_idx = self.df.loc[index, 'label_idx']
        return (self.df.loc[index, 'utterance']), torch.FloatTensor(label_onehot), label_idx
