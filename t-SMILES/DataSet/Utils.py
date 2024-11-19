import os
import threading
from networkx import exception
import torch

import random
import numpy as np
import re

import csv
import time
import math
import numpy as np
import warnings

import joblib
import pandas as pd
import h5py

from tqdm import tqdm, trange

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle as shuffle_data

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit import DataStructs
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from DataSet.STDTokens import CTokens
#from DataSet.STDTokens import STDTokens_Common, STDTokens_SMI_All
#from DataSet.STDTokens import STDTokens_ZincMoses, STDTokens_ZincSingle
#from DataSet.STDTokens import STDTokens_ChemblDouble,STDTokens_ChemblSingle

from Tools.DataConv import BCDataConv
from MolUtils.RDKUtils.Utils import RDKUtils

'''
from sklearn.preprocessing import OneHotEncoder
x = [[11, "Spain"], [22, "France"], [33, "Spain"], [44, "Germany"], [55, "France"]]
y = OneHotEncoder().fit_transform(x).toarray()
print(y)
[[1. 0. 0. 0. 0. 0. 0. 1.]
[0. 0. 0. 0. 1. 1. 0. 0.]]
'''
from DataSet.STDDataLoader import CSTDDataLoader

from DataSet.STDTokens import CTokens, STDTokens_Common 
from DataSet.STDTokens import STDTokens_ZincMoses, STDTokens_ZincSingle
from DataSet.STDTokens import STDTokens_ChemblDouble,STDTokens_ChemblSingle

from DataSet.Tokenlizer import Tokenlizer

#from DataSet.Utils import DSParam, SeqUtils
def DSParam():  #'dict' 
    return {
        'data_path'      : './DataSet/ChEMBL/chembl_10.smi',
        'tokens'        : None,
        'start_token'   :'<', 
        'end_token'     :'>', 
        'max_len'       : 120,
        'use_cuda'      : False, 
        'pad_symbol'    : ' ', 
        'seed'          : None,
        'batch_size'    : 1,
        'split'         : 0.,
        'delimiter'     : '\t',   #read_object_property_file
        'cols_to_read'  : [0],    #read_object_property_file
        'rows_to_read'  : -1,     #-1: read all rows
        'has_header'   : False,   #read_object_property_file
        }  


class SeqUtils:
    #def Str2IntTensor(str, tokens, pad = False, max_length = 100, pad_symbol = ' '):#, device = 'cpu'
    #    #print(string)
    #    #print(tokens)
    #    tokens = set(tokens) | set(pad_symbol)
        
    #    subtokens, wordarray = SeqUtils.split_regex(str)
    #    if len (set(subtokens) |tokens) > len(tokens):
    #        raise ValueError('subtokens not in total tokens!')

    #    nlen = len(wordarray)
    #    if pad and nlen > max_length:
    #        raise ValueError('max_length should be larger than length of string when padding is true!')
         
    #    if pad:
    #        for i in range(max_length - nlen):
    #            wordarray.append(pad_symbol)

    #    tensor = torch.zeros(len(wordarray)).long()

    #    for c in range(len(wordarray)):
    #        try:
    #            tensor[c] = tokens.index(wordarray[c])
    #        except:
    #            continue

    #    return tensor.clone(), nlen

    #def Seq2IntTensor(seqs, tokens, pad = False, max_length = 100, pad_symbol = ' ', flip = False):
    #    #items of input seqs are in the same length
    #    #input:[[Brc1ccccc1Cc1cccs1],[CCCCC(C)c1cccs1]]
    #    #output:[[21 23 1 ....5 8 9],[5 5..........4 3 8]]
    #    #if pad:
    #    #   seqs, lengths = SeqUtils.pad_sequences(seqs, max_length= 200)

    #    tensor = []
    #    lengths = []
    #    inted, sublen = SeqUtils.Str2IntTensor(seqs[0], tokens, pad = pad, max_length = max_length, pad_symbol = pad_symbol)
    #    tlen = len(inted)
    #    samelen = True
    #    for id, seq in enumerate(seqs):
    #        inted, nlen = SeqUtils.Str2IntTensor(seq, tokens, pad = pad, max_length = max_length, pad_symbol = pad_symbol)
    #        tensor.append(inted)
    #        lengths.append(nlen)

    #        if tlen != len(inted):
    #            samelen = False           
        
    #    if pad and not samelen:
    #        raise ValueError('all dataset shoule be in same length!')

    #    if samelen:
    #        tensor = torch.Tensor(tensor)
    #        if flip:
    #            tensor = np.flip(tensor, axis=1).copy()

    #    return tensor, tokens

    def count_tokens(smiles):
        if isinstance(smiles, str):
            smiles = [smiles]
        words = {}
        if isinstance(smiles, (list, tuple, dict, np.ndarray)):
            for sml in smiles:
                for c in sml:
                    if c in words:
                        words[c] +=1
                    else:
                        words[c] = 0

        print('words=',words)

        tokens = sorted(list(words.keys()))

        return tokens

    def prepross_smiles(smiles, remove_Aromatic = False):
        #tokens=set([''.join(list(x)) for x in self.smiles])   #no duplicate
        if isinstance(smiles, str):
            smiles = [smiles]

        smiles = list(set(smiles)) # remove duplicated   #74016->68624

        node_freq_dict = {}
        edge_freq_dict = {}        

        words = {}
        atoms = set([])
        bonds = set([])
        max_n_atoms = 0
        if isinstance(smiles, (list, tuple, dict, np.ndarray)):
            for i in trange(len(smiles), desc = 'SeqUtils.prepross_smiles......'):
                sml = smiles[i]
                try:
                    mol = Chem.MolFromSmiles(sml)
                    if mol is not None:
                        if remove_Aromatic:
                            RDKUtils.remove_Aromatic(mol)

                        sas = mol.GetAtoms()
                        sbs = mol.GetBonds()
                        if len(sas) > max_n_atoms:
                            max_n_atoms = len(sas)

                        for a in sas:
                            if a.GetSymbol() not in node_freq_dict:
                                node_freq_dict[a.GetSymbol()] = 0                        
                            node_freq_dict[a.GetSymbol()] += 1 
            
                        for b in sbs:
                            if b.GetBondType() not in edge_freq_dict:
                                edge_freq_dict[b.GetBondType()] = 0
                            edge_freq_dict[b.GetBondType()] += 1 

                        atoms = atoms.union(set(atom.GetSymbol() for atom in sas))
                        bonds = bonds.union(set(bond.GetBondType() for bond in sbs))

                    for c in sml:
                        if c in words:
                            words[c] +=1
                        else:
                            words[c] = 0
                except Exception as e:
                    #print(e.args)
                    pass

        tokens = sorted(list(words.keys()))
        atoms = list(atoms)
        bonds = list(bonds)        

        print('tokens=',tokens)
        print('atoms=',atoms)
        print('bonds=',bonds)
        return tokens, atoms, bonds, node_freq_dict, edge_freq_dict, max_n_atoms

    def read_smilesset(path):
        smiles_list = []
        with open(path) as f:
            for smiles in f:
                smiles_list.append(smiles.rstrip())

        return smiles_list

    def Str2IntTensor(string, tokens):#, device = 'cpu'
        #print(string)
        #print(tokens)
        tensor = torch.zeros(len(string)).long()

        for c in range(len(string)):
            try:
                tensor[c] = tokens.index(string[c])
            except:
                continue

        #tensor = tensor.to(device)
        return tensor.clone()

    def Seq2IntTensor(seqs, tokens, pad = False, flip = False, max_length= 200):
        #items of input seqs are in the same length
        #input:[[Brc1ccccc1Cc1cccs1],[CCCCC(C)c1cccs1]]
        #output:[[21 23 1 ....5 8 9],[5 5..........4 3 8]]
        if pad:
           seqs, lengths = SeqUtils.pad_sequences(seqs, max_length = max_length)

        tensor = np.zeros((len(seqs), len(seqs[0])))    #(1000, 77)
          
        for id, seq in enumerate(seqs):
            #print(seq)
            tensor[id] = SeqUtils.Str2IntTensor(seq, tokens)
            
        #for i in range(len(seqs)):
            #for j in range(len(seqs[i])):
            #    if seqs[i][j] in tokens:
            #        tensor[i, j] = tokens.index(seqs[i][j])
            #    else:
            #        tokens = tokens + [seqs[i][j]]
            #        tensor[i, j] = tokens.index(seqs[i][j])

        if flip:
            tensor = np.flip(tensor, axis=1).copy()

        return tensor, tokens

    def StrOHEncoder(str, token, flip = False):
        ct = CTokens(token)
        oh = ct.onehot_encode_single(str)

        return oh

    def Seq2OHTensor(smiles, token, flip = False): 
        ct = CTokens(token)
        oh = ct.onehot_encode(smiles)
        return oh

    def pad_sequences(seqs, max_length = None, pad_symbol = ' '):
        if max_length is None:
            max_length = -1
            for seq in seqs:
                max_length = max(max_length, len(seq))

        lengths = []
        pad_sep = []
        for i in range(len(seqs)):
            cur_len = len(seqs[i])
            if cur_len < max_length:
                lengths.append(cur_len)
                sp = seqs[i] + pad_symbol * (max_length - cur_len)
                pad_sep.append(sp)
            else:
                print(cur_len, seqs[i])

        return pad_sep, lengths

    def split_regex(str):
        tokens = set()   
        wordarray = []
        regex = '(\[[^\[\]]{1,6}\])'
        str = re.sub('\[\d+', '[', str)
        for word in re.split(regex, str):
            if word == '' or word is None:
                continue
            if word.startswith('['):
                wordarray.append(word)
            else:
                for i, char in enumerate(word):
                    wordarray.append(char)
                
        tokens.update(wordarray)    
        return tokens, wordarray

    def tokenize(seqs, tokens = None):
        #create new token based on smiles and input tokens
        if tokens is None:
            tokens = list(set(''.join(seqs)))
            tokens = list(np.sort(tokens))
            tokens = ''.join(tokens)

        token2idx = dict((token, i) for i, token in enumerate(tokens))
        num_tokens = len(tokens)

        return tokens, token2idx, num_tokens

    def read_smi_file(filename, unique=True, add_start_end_tokens=False):
        f = open(filename, 'r')
        molecules = []
        for line in f:
            if add_start_end_tokens:
                molecules.append('<' + line[:-1] + '>')
            else:
                molecules.append(line[:-1])
        if unique:
            molecules = list(set(molecules))
        else:
            molecules = list(molecules)
        f.close()
        return molecules, f.closed

    def read_smiles_csv(path):
        data = pd.read_csv(path, usecols=['SMILES'], squeeze=True).astype(str).tolist()
        return data

    def read_sp(path,  header = None, delimiter = ','):
        try:
            df = pd.read_csv(path, delimiter = delimiter,  header=header)
        except:
            df = pd.read_table(path)

        data_full = df.values.astype(np.float32)

        return data_full

    def read_object_property_file(path, 
                                  delimiter = ',', 
                                  cols_to_read = [0, 1],
                                  has_header = True, 
                                  return_type = 'default', #['dim1', 'str', 'default']    
                                  rows_to_read = -1,
                                  in_order = False,
                                  **kwargs):
        print(path)
        try:
            df = pd.read_csv(path, delimiter = delimiter, header = None)
        except:
            df = pd.read_table(path)

        data_full = df.values

        #f = open(path, 'r')
        #reader = csv.reader(f, delimiter = delimiter)
        #data_full2 = np.array(list(reader))
        #f.close()

        if has_header:
            start_position = 0
        else:
            start_position = 0

        assert len(data_full) > start_position

        #data = [[] for _ in range(len(cols_to_read))]
        shape = data_full.shape
        r = shape[0]
        if data_full.ndim == 1:
            c = 1
        else:
            c = shape[1]
            
        data = []
        if len(cols_to_read) == 1 and cols_to_read[0] == -1:  #read all data as one list
            data = np.array(data_full).flatten()
        else:
            for i in range(len(cols_to_read)):
                col = cols_to_read[i]
                if col >=0 and col < c:
                    data.append(data_full[:,col])
                    #data[:i] = data_full[start_position:, col]
            data = np.array(data).T
                
        if rows_to_read > 0 and rows_to_read < len(data):   #return the fist size rows
            if in_order:
                data = data[start_position:rows_to_read,:]
            else:
                rows_to_read = rows_to_read if rows_to_read < len(data) else len(data)
                sels = np.random.randint(0, len(data) - 1, rows_to_read)
                sub_data = [data[i] for i in sels]
                data = sub_data


        if return_type == 'dim1':   #return [*, *, *]
            if len(cols_to_read) == 1:
                data = data.reshape(len(data))
        elif return_type == 'str':   #if true, return ['', ''] else return [[''],['']]
            sdata = []
            if len(cols_to_read) == 1:
                for i in range(len(data)):
                    if isinstance(data[i], str):
                        sdata.append(data[i])
                    else: 
                        sdata.append(data[i][0])
            else:
                for i in range(len(data)):
                    idata = data[i]
                    d = []
                    for j in range(len(idata)):
                        if isinstance(idata[j], str):
                            d.append(idata[j])
                        else: 
                            d.append(idata[j][0])
                    sdata.append(d)

            data = np.array(sdata)
        elif return_type == 'float': #return[[float],[float]]
            data = data.astype(np.float32)
        elif return_type == 'int': #return[[float],[float]]
            data = data.astype(np.int32)
            #data = data.astype(np.int64)
        else:
            data = data     #return  [[''],['']], dtype = 'O'

        print(f'Loading data done! length = {len(data)}')
        return data
        
    def load_h5f(filename, split = True):
        h5f = h5py.File(filename, 'r')

        if split:
            data_train = h5f['data_train'][:]   #+		shape	(40000, 120, 33)	tuple
        else:
            data_train = None

        data_test = h5f['data_test'][:] #+		shape	(10000, 120, 33)	tuple
        charset =  h5f['charset'][:] #+		shape	(33,)	tuple
        h5f.close()
  
        save = False
        if save:
           path = os.path.dirname(filename)
           
           #df = pd.DataFrame(data_train)
           #df.to_csv(os.path.join(path,'train.csv'), index = False, header = False, na_rep="NULL")
         
           #df = pd.DataFrame(data_test)
           #df.to_csv(os.path.join(path,'test.csv'), index = False, header = False, na_rep="NULL")
           
           #df = pd.DataFrame(BCDataConv.S12str(x) for x in charset)
                   #return "".join(map(lambda x: charset[x], vec)).strip()

           sc = map(lambda x: BCDataConv.S12str(x), charset)
           df = pd.DataFrame(sc)
           df.to_csv(os.path.join(path,'charset.csv'), index = False, header = False, na_rep="NULL")

        if split:
            return (data_train, data_test, charset)
        else:
            return (data_test, charset)

    def ClsLabelOHEncoder(labels, flip = False):
        #token = {}
        #for a  in labels:
        #    token[a] +=1 
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        integer_encoded = label_encoder.transform(labels)

        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        #inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
        return onehot_encoded


    def LabelOHEncoder(arraydata, token, flip = False):
        #not tested
        #data = ['Python', 'Java', 'Python', 'Python', 'C++', 'C++', 'Java', 'Python', 'C++', 'Java' ]
        #vals = np.array(data)
        #output:[2 1 2 2 0 0 1 2 0 1]
        label_encoder = LabelEncoder()
        label_encoder.fit(token)
        integer_encoded = label_encoder.transform(arraydata)
        integer_encoded = label_encoder.fit_transform(arraydata)
        print(integer_encoded)
       
        oh_encoder = OneHotEncoder()
        oh_encoder.fit([[1],[2],[3],[4]])
        ohe.transform([[2],[3],[1],[4]]).toarray()

        return integer_encoded

    def LoadSmiles(file, length):
        print('loading SMILES...')
        smiles=[]
        with open(file) as f: 
            for s in f:
                if len(s.rstrip()) > 0:
                   smiles.append(s.rstrip())
                   #print(len(smiles))

        if length > 0:
            length = min(length,len(smiles))
            smiles = smiles[:length]
        print('Reading done.')
        
        f.close()

        return smiles
    
    def SaveSmiles(file, smiles):
        if unique:
            smiles = list(set(smiles))
        else:
            smiles = list(smiles)

        f = open(file, 'w')
        for mol in smiles:
            f.writelines([mol, '\n'])

        f.close()
        return f.closed

    def load_smiles_data(file, cv = False, normalize_y=True, 
                         k=5, header=0, index_col=0, 
                         delimiter=',', x_y_cols=(0, 1),
                         reload=True, seed=None, verbose=True, 
                         shuffle=0, create_val=True, train_size=0.8):
        assert (os.path.exists(file)), f'File {file} cannot be found.'
        assert (0 < train_size < 1), 'Train set size must be between (0,1)'

        def log(t):
            if verbose:
                print(t)

        data_dict = {}
        transformer = None

        data_dir, filename = os.path.split(file)

        suffix = '_cv' if cv else '_std'
        save_dir = os.path.join(data_dir, filename.split('.')[0] + f'_data_dict{suffix}.joblib')
        trans_save_dir = os.path.join(data_dir, filename.split('.')[0] + f'_transformer{suffix}.joblib')

        # Load data if possible
        if reload and os.path.exists(save_dir):
            log('Loading data...')
            with open(save_dir, 'rb') as f:
                data_dict = joblib.load(f)
                transformer = None

            if os.path.exists(trans_save_dir):
                with open(trans_save_dir, 'rb') as f:
                    transformer = joblib.load(f)

            log('Data loaded successfully')
            return data_dict, transformer

        # Read and process data
        dataframe = pd.read_csv(file, header=header, index_col=index_col, delimiter=delimiter)

        if shuffle > 0:
            for i in range(shuffle):
                dataframe = shuffle_data(dataframe)

        log(f'Loaded data size = {dataframe.shape}')
        X = dataframe[dataframe.columns[x_y_cols[0]]].values
        y = dataframe[dataframe.columns[x_y_cols[1]]].values.reshape(-1, 1)

        if normalize_y:
            log('Normalizing labels...')
            transformer = StandardScaler()
            y = transformer.fit_transform(y)

        log(f'Data directory in use is {data_dir}')

        # Split data
        if cv:
            log(f'Splitting data into {k} folds. Each fold has train, val, and test sets.')
            cv_split = KFold(k, shuffle=True, random_state=seed)
            for i, (train_idx, test_idx) in enumerate(cv_split.split(X, y)):
                x_train, y_train = X[train_idx], y[train_idx]

                if create_val:
                    x_val, x_test, y_val, y_test = train_test_split(X[test_idx], y[test_idx], test_size=.5,
                                                                    random_state=seed)
                    data_dict[f'fold_{i}'] = {'train': (x_train, y_train),
                                              'val': (x_val, y_val),
                                              'test': (x_test, y_test)}
                else:
                    data_dict[f'fold_{i}'] = {'train': (x_train, y_train),
                                              'test': (X[test_idx], y[test_idx])}
            log('CV splitting completed')
        else:
            log('Splitting data into train, val, and test sets...')
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size, random_state=seed)
            data_dict['train'] = (x_train, y_train)

            if create_val:
                x_val, y_val, x_test, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=seed)
                data_dict['val'] = (x_val, y_val)

            data_dict['test'] = (x_test, y_test)
            log('Splitting completed.')

        # Persist data if allowed
        if reload:
            with open(save_dir, 'wb') as f:
                joblib.dump(dict(data_dict), f)

            if normalize_y:
                with open(trans_save_dir, 'wb') as f:
                    joblib.dump(transformer, f)

        return data_dict, transformer

    def GetTokens(smiles):
        from OpenChem.openchem.data.utils import get_tokens

        tokens, token2idx, num_tokens = get_tokens(smiles)   #less chars
        print(tokens)

        return tokens, token2idx, num_tokens

class OHEncoder(nn.Module):
    def __init__(self, vocab_size, return_tuple=False, device='cpu'):
        super(OHEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.device = device
        self.return_tuple = return_tuple

    def forward(self, inp):
        is_list = False
        if isinstance(inp, (tuple, set, list)):
            x = inp[0]
            is_list = True
        else:
            x = inp
        x = torch.nn.functional.one_hot(x, self.vocab_size).float().to(self.device)
        x = x.permute(1, 0, 2)
        if self.return_tuple:
            if not is_list:
                inp = [None]
            inp[0] = x
            return inp
        return x

class Test():
    def test_CToken():
        print('---------------test_CToken----------------------------------')
        file = r'G:\ProjectTF\BioChemoTCH\Dataset\Zinc\250k_rndm_zinc_drugs_clean_3.smi'
        smiles = SeqUtils.LoadSmiles(file, 5)
        print(smiles)
    
        #token = CSTDTokens.ZincTokens()
        token = CSTDTokens.ZincSingleTokens()
        #token = CSTDTokens.CommonDoubleTokens()
        #token = CSTDTokens.ChemblSingleTokens()
        #token = CSTDTokens.ChemblDoubleTokens()

        ct = CTokens(token, includeSE = True)
        oh = ct.onehot_encode(smiles)
        print('oh=',oh)
        inverted = ct.onehot_decode(oh)
        print('inverted=',inverted)
     
        oh = ct.int_encode(smiles, is_pad = ct.is_pad, split = True)
        print('int_oh=',oh)
        inverted = ct.int_decode(oh)
        print('int_inverted=',inverted)

        return

    def test_Str2IntTensor():
        print('---------------test_Str2IntTensor----------------------------------')
        token = CSTDTokens.ZincTokens()
        token = CSTDTokens.ZincSingleTokens()
        token = CSTDTokens.CommonDoubleTokens()
        token = CSTDTokens.ChemblSingleTokens()
        token = CSTDTokens.ChemblDoubleTokens()

        print("Tokens=", token)
       
        sml = 'C[C@H]1CCCN(c2ccc(C(=O)Nc3ccc(N4CCOCC4)cc3)cc2[N+](=O)[O-])C1'
   
        res = SeqUtils.Str2IntTensor(sml, token)
        print(res)

        return

    def test_Seq2IntTensor():
        print('---------------test_Seq2IntTensor----------------------------------')
        #token = CSTDTokens.ZincTokens()
        #token = CSTDTokens.ZincSingleTokens()
        token = CSTDTokens.CommonDoubleTokens()
        #token = CSTDTokens.ChemblSingleTokens()
        #token = CSTDTokens.ChemblDoubleTokens()

        file = r'D:\ProjectTF\BioChemoTCH\DataSet\Zinc\ZINC_10.txt'
        seqs = SeqUtils.read_object_property_file(file, '\t', [0])
        print(seqs)

        res, tks = SeqUtils.Seq2IntTensor(seqs, token, pad = True, max_length = 100, pad_symbol = ' ', flip = False)
       
        print(res)
       
        return
    
    def test_StrOHEncoder():
        print('---------------test_StrOHEncoder----------------------------------')
        token = CSTDTokens.ZincTokens()
        token = CSTDTokens.ZincSingleTokens()
        token = CSTDTokens.CommonDoubleTokens()
        token = CSTDTokens.ChemblSingleTokens()
        token = CSTDTokens.ChemblDoubleTokens()

        print("Tokens=", token)
       
        sml = 'C[C@H]1CCCN(c2ccc(C(=O)Nc3ccc(N4CCOCC4)cc3)cc2[N+](=O)[O-])C1'   
        res = SeqUtils.StrOHEncoder(sml, token)
        print(res)
        return

    def test_Seq2OHTensor():
        print('---------------test_Seq2OHTensor----------------------------------')
        token = CSTDTokens.ZincTokens()
        token = CSTDTokens.ZincSingleTokens()
        token = CSTDTokens.CommonDoubleTokens()
        token = CSTDTokens.ChemblSingleTokens()
        token = CSTDTokens.ChemblDoubleTokens()
        
        file = r'D:\ProjectTF\BioChemoTCH\DataSet\ChEMBL\chembl_10.smi'
        smiles = SeqUtils.read_object_property_file(file, '\t', [0])

        res = SeqUtils.Seq2OHTensor(smiles, token)
        print(res)

        return

    def test_pad_sequences():
        print('---------------test_pad_sequences----------------------------------')
        sml = 'C[C@H]1CCCN(c2ccc(C(=O)Nc3ccc(N4CCOCC4)cc3)cc2[N+](=O)[O-])C1'   
        sml2 = 'CCO[C@H]1C(=O)O[C@H]([C@@H](O)CO)C1=O'   
        smiles = [sml,sml2]
        seqs, lengths = SeqUtils.pad_sequences(smiles, max_length= 100)
        print(seqs)
        print(lengths)

        return
    
    def test_split_regex():
        print('---------------test_tokenize----------------------------------')
        token = CSTDTokens.ZincTokens()
        token = CSTDTokens.ZincSingleTokens()
        token = CSTDTokens.CommonDoubleTokens()
        token = CSTDTokens.ChemblSingleTokens()
        token = CSTDTokens.ChemblDoubleTokens()
        token = []

        print("Tokens=", token)
       
        sml = 'C[C@H]1CCCN(c2ccc(C(=O)Nc3ccc(N4CCOCC4)cc3)cc2[N+](=O)[O-])C1'   

        tokens, wordarray = SeqUtils.split_regex(sml)
        print(tokens)
        print(wordarray)
        return

    def test_tokenize():
        print('---------------test_tokenize----------------------------------')
        token = CSTDTokens.ZincTokens()
        token = CSTDTokens.ZincSingleTokens()
        token = CSTDTokens.CommonDoubleTokens()
        token = CSTDTokens.ChemblSingleTokens()
        token = CSTDTokens.ChemblDoubleTokens()

        print("Tokens=", token)
       
        sml = 'C[C@H]1CCCN(c2ccc(C(=O)Nc3ccc(N4CCOCC4)cc3)cc2[N+](=O)[O-])C1'   

        tokens, token2idx, num_tokens = SeqUtils.tokenize(sml, token)
        print(tokens)
        print(token2idx)
        print(num_tokens)
        return

    def test_read_smi_file():
        print('---------------test_read_smi_file----------------------------------')

        return

    def test_read_object_property_file():
        print('---------------test_read_object_property_file----------------------------------')
        file = r'D:\ProjectTF\BioChemoTCH\DataSet\ChEMBL\chembl_10.smi'
        smiles = SeqUtils.read_object_property_file(file, '\t', [0])
        #smiles = CSTDDataLoader.LoadSmiles(file, 5)
        print(smiles)

        return

    def test_LabelOHEncoder():
        print('---------------test_LabelOHEncoder----------------------------------')
        return

    def test_SaveSmiles():
        print('---------------test_SaveSmiles----------------------------------')
        return

    def test_load_smiles_data():
        print('---------------test_load_smiles_data----------------------------------')
        return

    def test_GetTokens():
        print('---------------test_GetTokens----------------------------------')
        file = r'G:\ProjectTF\BioChemoTCH\Dataset\Zinc\250k_rndm_zinc_drugs_clean_3.smi'
        #tokens35:'#()+-/12345678=@BCFHINOPS[\\]clnors '
        #file = r'G:\ProjectTF\BioChemoTCH\Dataset\ChEMBL\chembl_21_1576904.smi'
        #token64:'\t #%()+,-./0123456789=ABCEFHILMNOPRST[\\]abcdefhilmnopqrstuxyz{} '
        smiles = SeqUtils.LoadSmiles(file,0)
        tokens, token2idx, num_tokens = SeqUtils.GetTokens(smiles)
        print(tokens)
        return

if __name__ == '__main__':
    #import sys
    #sys.path.append('./OpenChem')
   
    #label=[2, 1, 2, 2, 0, 0, 1, 2, 0, 1]
    #oh = SeqUtils.ClsLabelOHEncoder(label)

    #Test.test_CToken()
    #Test.test_Str2IntTensor()
    #Test.test_Seq2IntTensor()
    #Test.test_StrOHEncoder()
    #Test.test_Seq2OHTensor()
    #Test.test_pad_sequences()
    #Test.test_split_regex()
    #Test.test_tokenize()
    #Test.test_read_object_property_file()
    #Test.test_GetTokens()

           
    ctokens = CTokens(STDTokens_SMI_All(), invalid= True)
    sml = '<CC(C(=O)O)n1[se]c2ccccc2c1=O'
    sml = '<$C=C1C(=O)OC2/C=C(\\C)CC(O)/C=C(\\C)CC(OC(=O)/C(=C\\C)COC(C)=O)C12'
    tks = ctokens.tokenlizer.tokenize(sml)
    print(tks)
