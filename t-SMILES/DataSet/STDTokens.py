from builtins import str
import numpy as np

from DataSet.Tokenlizer import Tokenlizer
from DataSet.JTNN.MolTree import Vocab

from enum import Enum
class TokenEncoder(Enum):
    TE_Single = 0
    TE_Double = 1
    TE_Multiple = 2

class CTokens:
    def __init__(self, stdtokens, exter_tokens = None, is_pad = False, pad_symbol = ' ', startend = True, 
                 max_length = 120,  flip = False, invalid = False, onehot = True):
        self.STDTokens = stdtokens
        self.code_type = stdtokens.Encoder()
        self.tokens = stdtokens.Tokens()
        
        self.max_length = max_length

        self.start_token = '<'
        self.end_token = '>'
        self.pad_symbol = pad_symbol  
        self.pad_index = 0
        self.invalid_token = '&'

        self.is_pad = is_pad   
        self.flip = flip
        self.startend = startend
        self.flag_tokens = [self.pad_symbol, self.start_token, self.end_token]

        if self.pad_symbol not in self.tokens:
           self.tokens.insert(0, self.pad_symbol)
           
        if startend:
            if self.start_token not in self.tokens:
               self.tokens.insert(1, self.start_token)

            if self.end_token not in self.tokens:
                self.tokens.insert(2, self.end_token)

        if exter_tokens is not None:
            for t in exter_tokens:
                self.tokens.append(t) 
            self.code_type = TokenEncoder.TE_Multiple

        if invalid:
            if self.invalid_token not in self.tokens:
                self.tokens.append(self.invalid_token)

        self.n_tokens = len(self.tokens)
         
        self.table_2_chars = list(filter(lambda x: len(x) > 1, self.tokens))
        self.table_1_chars = list(filter(lambda x: len(x) == 1, self.tokens))
        self.exter_tokens = exter_tokens

        self.onehot_dict = {}
        self.char2int = {}
        self.int2char = {}
        for i, symbol in enumerate(self.tokens):
            self.char2int[symbol] = i
            self.int2char[i] = symbol

        if onehot:
            for i, symbol in enumerate(self.tokens):
                vec = np.zeros(self.n_tokens, dtype = np.float32)
                vec[i] = 1
                self.onehot_dict[symbol] = vec          
        
        if startend:
            self.start_index  = self.char2int[self.start_token]
            self.end_index    = self.char2int[self.end_token]
        else:
            self.start_index  = -1
            self.end_index = -1

        if invalid:
            self.invalid_index = self.char2int[self.invalid_token]
        else:
            self.invalid_index = -1

        self.sorted_token = sorted(self.tokens, key = lambda i:len(i), reverse=True)
        self.tokenlizer = Tokenlizer(self.tokens, invalid_token = self.invalid_token)
        return
  
    def get_index(self, symbol):
        if symbol in self.tokens:
            return self.char2int[symbol]
        else:
            return -1

    def get_token(self, idx):
        idx = int(idx)
        if idx < self.n_tokens:
            return self.int2char[idx]
        else:
            return

#end CTokens    

class STDTokens:
    def __init__(self, *args, **kwargs):
        pass

    def Encoder(self):
        pass

    def Tokens(self):
        pass


class  STDTokens_Frag_File(STDTokens):
    def __init__(self, voc_file, add_std = False, *args, **kwargs):
        std_tokens = [' ']
        if voc_file is None:
            tokens = ['C']
        elif isinstance(voc_file, str):
            tokens = [x.strip("\r\n ") for x in open(voc_file)]
        elif isinstance(voc_file, (list, np.array, tuple)):
            tokens = []
            for f in voc_file:
                tokens.extend([x.strip("\r\n ") for x in open(f)])
        else:
             tokens = ['C']

        if add_std:
            tokens = list(set(tokens).union(set(std_tokens)))           
        else:
            tokens = list(set(tokens))

        tokens.sort()     
        self.tokens = tokens

        maxlen = 1
        for t in tokens:
            if len(t) > maxlen:
                maxlen = len(t)

        if maxlen == 1:
            self.token_encoder = TokenEncoder.TE_Single
        elif maxlen == 2:
            self.token_encoder = TokenEncoder.TE_Double
        else:
            self.token_encoder = TokenEncoder.TE_Multiple
                 
        self.vocab = Vocab(self.tokens)

        self.tokens_encode = dict((token, i) for i, token in enumerate(self.tokens))
        self.tokens_decode = dict((i, token) for i, token in enumerate(self.tokens))

        return

    def Encoder(self):
        return self.token_encoder

    def Tokens(self):
        return self.tokens

    def get_index(self, token):
        return self.tokens_encode[token]

    def get_token(self, idx):
        return self.tokens_decode[int(idx)]