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

    df = pd.read_csv(f'./data/{dataset_name}.csv', sep="|")
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


if __name__ == '__main__':
    import re

    def processing(dataset_name :str,  val_frac, test_frac):

        df = pd.read_csv(f'./data/{dataset_name}.csv', sep="|")
        # sort by length
        df = df.sort_values(by="utterance", key=lambda x: x.str.len()).reset_index(drop=True)
        print(f"==================== {dataset_name} loaded ====================")
        
        labels = list(df['intent'].unique())
        if 'no_intent' in labels:
            df = df.loc[df.loc[:,'intent'] != 'no_intent'].reset_index().drop(columns='index')
            labels = list(df['intent'].unique())

        df['label_idx'] = pd.factorize(df['intent'])[0]
        return df
    
    def unique_word(df):
        uts = set()
        lens = []
        for i, utter in enumerate(df.utterance):
            utter = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", utter)
            df.loc[i, 'utterance'] = utter
            uts.update(utter.split())
            lens.append(len(utter.split()))
            # df.loc[i, 'len'] = len(utter.split())
        # df['len'] = df.utterance.apply(lambda x: len(x.split()))
        # uniq = list(df["unique_word"].sum())
        # print(uniq)
        return uts, lens

    atis_tr = processing('ATIS', val_frac=0.2, test_frac=0.1)
    bank_tr = processing('BANKING77', val_frac=0.2, test_frac=0.1)
    clinc_tr =processing('clinc150', val_frac=0.2, test_frac=0.1)
    hwu_tr = processing('HWU64', val_frac=0.2, test_frac=0.1)
    mcid_tr =processing('mcid', val_frac=0.2, test_frac=0.1)
    rest_tr =processing('restaurant', val_frac=0.2, test_frac=0.1)

    trs = []
    trs.extend((atis_tr, bank_tr, clinc_tr, hwu_tr, mcid_tr, rest_tr))



    for i, tr in enumerate(trs):
        print('\n\n')
        print("unique intents:      ", len(tr.intent.unique()),'\n', tr.intent.unique())
        print()
        print("utterances           ", len(tr))
        print()
        uts, lens = unique_word(tr)
        print("unique words:        ", len(uts))  
        print()
        print("shortest utterance:   ",  min(lens), tr.iloc[lens.index(min(lens))].utterance,'\n', len(tr.iloc[0].utterance.split()), tr.iloc[0].utterance)
        print()
        print("longest utterance:   ",  max(lens), tr.iloc[lens.index(max(lens))].utterance, '\n', len(tr.iloc[-1].utterance.split()), tr.iloc[-1].utterance)
        print()
        print("average utterance:   ",  sum(lens)/len(lens))
        print('\n')