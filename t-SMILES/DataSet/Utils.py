import numpy as np
import pandas as pd
import torch

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

        if flip:
            tensor = np.flip(tensor, axis=1).copy()

        return tensor, tokens



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

    def tokenize(seqs, tokens = None):
        #create new token based on smiles and input tokens
        if tokens is None:
            tokens = list(set(''.join(seqs)))
            tokens = list(np.sort(tokens))
            tokens = ''.join(tokens)

        token2idx = dict((token, i) for i, token in enumerate(tokens))
        num_tokens = len(tokens)

        return tokens, token2idx, num_tokens


    def read_object_property_file(path,
                                  delimiter = ',', 
                                  cols_to_read = [0, 1],
                                  has_header = True, 
                                  return_type = 'default', #['dim1', 'str', 'default']    
                                  rows_to_read = -1,
                                  in_order = False,
                                  **kwargs):
        try:
            df = pd.read_csv(path, delimiter = delimiter)
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
        else:
            data = data     #return  [[''],['']], dtype = 'O'

        print(f'Loading data done! length = {len(data)}')
        return data


