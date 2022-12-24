import threading
import torch

import random
import numpy as np


from torch.utils.data import Dataset


from DataSet.Utils import DSParam, SeqUtils
from DataSet.STDTokens import STDTokens, CTokens, EncoderType, STDTokens_Customed

class DataGenerator(object):
    def __init__(self, 
                 data_path, 
                 raw_data = None,
                 ctokens = None,
                 tokens = None,
                 use_cuda = False, 
                 seed = None,
                 batch_size = 1,
                 cols_to_read = [0],
                 rows_to_read = -1,
                 encoder_type = EncoderType.ET_Int,
                 has_header = True,
                 **kwargs):
        super(DataGenerator, self).__init__()

        if seed:
            np.random.seed(seed)

        self.data_path = data_path
        self.batch_size = batch_size
        self.encoder_type = encoder_type
 
        if ctokens is None:
            stdtoken = STDTokens_Customed(tokens)
            ctokens = CTokens(stdtoken)
           
        self.ctoken = ctokens
        self.tokens = ctokens.tokens

        self.start_token = ctokens.start_token
        self.end_token = ctokens.end_token
        self.pad_symbol = ctokens.pad_symbol

        self.start_index = ctokens.start_index
        self.end_index = ctokens.end_index
        self.pad_index = ctokens.pad_index

        if 'tokens_reload' in kwargs:
            self.tokens_reload = kwargs['tokens_reload']

        if data_path is not None:
            data = SeqUtils.read_object_property_file(data_path,  
                                                      cols_to_read = cols_to_read, 
                                                      rows_to_read = rows_to_read, 
                                                      has_header = has_header,
                                                      return_type = 'str',
                                                      **kwargs)
        else:
            data = raw_data

        if data is None:
            return None

        self.samples = []
        for i in range(len(data)):
            if len(data[i].strip()) <= self.ctoken.max_length-2:
                sample = data[i].strip()

                if self.ctoken.startend:
                    sample = (self.ctoken.start_token + sample + self.ctoken.end_token)

                self.samples.append(sample)

        self.samples_len = len(self.samples)
        self.all_characters, self.char2idx, self.n_characters = SeqUtils.tokenize(self.samples, self.tokens)

        dif = (list(set(self.all_characters).difference(set(ctokens.tokens))))
        dif1 = (list(set(dif).difference(set(ctokens.flag_tokens))))      

        if len(dif1) > 0 :
            print('Tokens are different, some samples will be ignored when analysing!')
            print('Different tokens are:', dif)
               
        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        

        print('Data_File=',self.data_path)
        print('sample_len=',self.samples_len)

        return 

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def random_chunk_batch(self, batch_size = None, index = None):
        batch_size = batch_size if batch_size < self.samples_len else self.samples_len
        
        if index is None:
            index = np.random.randint(0, self.samples_len - 1, batch_size)

        return [self.samples[i][:-1] for i in index], [self.samples[i][1:] for i in index]
    
    def random_training_set_smiles(self, batch_size = None, index = None):
        if batch_size is None:
            batch_size = self.batch_size

        assert (batch_size > 0)

        batch_size = batch_size if batch_size < self.samples_len else self.samples_len

        if index is None:
            index = np.random.randint(0, self.samples_len - 1, batch_size)

        #return [self.samples[i][1:-1] for i in index]   #remove the first and last one
        return [self.samples[i] for i in index]

    def random_training_set_batch(self, batch_size = None, return_seq_len = False,
                                  RNN_Circle = True, return_type = 'long', rows = None):
        t = threading.currentThread()
        #print(f'\n random_training_set_batch, Thread name:{t.getName()}, Thread id : {t.ident}')

        if batch_size is None or batch_size == 0:
            batch_size = self.batch_size
                
        batch_size = batch_size if batch_size < self.samples_len else self.samples_len
        assert (batch_size > 0)

        if rows is None:
            if RNN_Circle:  #inp = '<****', target = '****>'
                inp, target = self.random_chunk_batch(batch_size)
            else:
                inp = self.random_training_set_smiles(batch_size)
                target = inp
        else:
            if RNN_Circle:  #inp = '<****', target = '****>'
                inp, target = self.random_chunk_batch(batch_size, rows)
            else:
                inp = self.random_training_set_smiles(batch_size, rows)
                target = inp

        if self.ctoken.is_pad == False:
            inp, inp_seq_len = self.ctoken.pad_sequences(inp)
            target, target_seq_len = self.ctoken.pad_sequences(target)

        if self.encoder_type == EncoderType.ET_Int:
            inp_tensor, inp_seq_len         = self.ctoken.int_encode(inp, is_pad = self.ctoken.is_pad)
            target_tensor, target_seq_len   = self.ctoken.int_encode(target, is_pad = self.ctoken.is_pad)
        else:
            inp_tensor, inp_seq_len         = self.ctoken.int_encode(inp, is_pad = self.ctoken.is_pad)
            target_tensor, target_seq_len   = self.ctoken.int_encode(target, is_pad = self.ctoken.is_pad)

        self.n_characters = len(self.all_characters)

        if return_type == 'long':
            inp_tensor = torch.tensor(inp_tensor).long()
            target_tensor = torch.tensor(target_tensor).long()
        else:
            inp_tensor = torch.tensor(inp_tensor).float()
            target_tensor = torch.tensor(target_tensor).float()

        if self.use_cuda:
            inp_tensor = inp_tensor.to('cuda:0')
            target_tensor = target_tensor.to('cuda:0')

        if return_seq_len:
            return inp_tensor, target_tensor, (inp_seq_len, target_seq_len)

        return inp_tensor, target_tensor
