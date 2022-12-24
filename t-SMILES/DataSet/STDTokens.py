from builtins import str
import numpy as np
import re

import torch

from DataSet.Tokenlizer import Tokenlizer
from DataSet.JTNN.MolTree import Vocab

from enum import Enum, unique
class TokenEncoder(Enum):
    TE_Single = 0
    TE_Double = 1
    TE_Multiple = 2

class EncoderType(Enum):
    ET_Str      = 0
    ET_Int      = 1
    ET_OH       = 2
    ET_Morgan   = 3    #len = 167
    ET_MACCS    = 4
    ET_Coulomb  = 5
    ET_Hamiltonian = 6
    ET_Graph    = 7

class CTokens:
    def __init__(self, stdtokens, exter_tokens = None, is_pad = False, pad_symbol = ' ', startend = True, 
                 max_length = 120,  flip = False, invalid = False):
        self.code_type = stdtokens.Encoder()
        self.tokens = stdtokens.Tokens()
        
        self.max_length = max_length

        self.start_token = '<'
        self.end_token = '>'
        self.pad_symbol = pad_symbol   #' '
        self.pad_index = 0
        self.invalid_token = '&'

        self.is_pad = is_pad    #is_pad
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
        #self.char2int = dict((token, i) for i, token in enumerate(tokens)) 
        #self.int2char = dict(( i, token) for i, token in enumerate(tokens))
        for i, symbol in enumerate(self.tokens):
            vec = np.zeros(self.n_tokens, dtype = np.float32)
            vec[i] = 1
            self.onehot_dict[symbol] = vec          
            self.char2int[symbol] = i
            self.int2char[i] = symbol
        
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

    def get_token_by_oh(self, oh):
        idx = np.argmax(oh)
        return self.get_token(idx)

    def array_to_tensor(self, seqs):
        tensor = [torch.tensor(x, dtype = torch.long, device = 'cpu') for x in seqs]
        return tensor
    
    def onehot_tensor(self):
        tensor = []
        for i in range(len(self.onehot_dict)):
            tensor.append(self.onehot_dict[self.int2char[i]])
        tensor = torch.tensor(tensor).float()
        return tensor
    
    def remove_start_end(self, strarray):
        if self.startend:
            result = []
            for str in strarray:
                res = self.remove_start_end_single(str)
                result.append(res)

            return result
        else:
            return strarray

    def remove_start_end_single(self, str):
        res = re.findall(r'[<](.*?)[>]', str) 
        if len(res) > 0 :
            return res[0]
        else:
            return str

    def split_regex(self, str):
        tokens = set()   
        wordarray = []
        regex = '(\[[^\[\]]{1,6}\])'
        #str = re.sub('\[\d+', '[', str)
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

    def split(self, smiles):  #organize input smiles to valid tokened string
        #double token
        #smiles = smiles + ' '
        N = len(smiles)
        tokend = []
        i = 0
        while (i < N-1):
            c1 = smiles[i]
            c2 = smiles[i:i + 2]

            if c2 in self.table_2_chars:
                tokend.append(c2)
                i += 2
                continue

            if c1 in self.table_1_chars:
                tokend.append(c1)
                i += 1
                continue

            i += 1
         
        if i == N-1:
            tokend.append(smiles[N-1])

        return tokend
    
    def tokenlize(self, str):
        tokenized_str = self.tokenlizer.tokenize(inseq = str)

        return tokenized_str

    def onehot_encode_single(self, instr, is_pad= False): #convert one smiles to onehot code
        try:
            if isinstance(instr, (str)):
                tokenized_str = self.tokenlize(instr)
            else:
                tokenized_str = instr
            
            if is_pad:
                tokenized_str, nlen = self.pad_str(tokenized_str)
            else:
                nlen = len(tokenized_str)            

            result = np.array([self.onehot_dict[symbol] for symbol in tokenized_str], dtype = np.float32)
        except:
            result = []
        #result = result.reshape(1, result.shape[0], result.shape[1])
        return result, nlen

    def onehot_encode(self, seqs, is_pad= False): #convert tokenized_smiles array[] to onehot arrar [][]
        result = []
        lengths = []

        for str in seqs:
            res, nlen = self.onehot_encode_single(str, is_pad = is_pad)
            if len(res) > 0:
                result.append(res)
                lengths.append(nlen)

        return result, lengths

    def onehot_encode_tensor(self, seqs, is_pad= False):
        result,lengths = self.onehot_encode(seqs, is_pad)

        result = torch.tensor(np.asarray(result), dtype=torch.long, device='cpu')
        lengths = torch.tensor(np.asarray(lengths), dtype=torch.long, device='cpu')

        return result, lengths

    def onehot_decode_single(self, onehot):
        onehot = np.array(onehot)
        token = ''
        for oh in onehot:
            pos = np.argmax(oh)
            token += self.int2char[pos]

        return token
    
    def onehot_decode(self, onehotarray):
        result = []
        for oh in onehotarray:
            res = self.onehot_decode_single(oh)
            if len(res) > 0:
                result.append(res)
                
        return result
    
    def onehot_to_int_single(self, onehot):
        onehot = np.array(onehot)
        token = []
        for oh in onehot:
            pos = np.argmax(oh)
            token.append(pos)

        return token
    
    def onehot_to_int(self, onehotarray):
        result = []
        for oh in onehotarray:
            res = self.onehot_to_int_single(oh)
            if len(res) > 0:
                result.append(res)

        return result

    def onehot_to_FP(self, onehotarray, encoder_type):
        seqs = self.onehot_decode(onehotarray)
        #seqs = [s.strip() for s in seqs]

        if encoder_type == EncoderType.ET_Morgan:
            fps, lengths, valid_smls = self.morgan_encode(seqs)
        elif encoder_type == EncoderType.ET_MACCS:
            fps, lengths, valid_smls = self.maccs_encode(seqs)
        
        valid_oh, _ = self.onehot_encode(valid_smls, pad = False)
        
        fp_tensor = torch.tensor(fps).float()
        oh_tensor = torch.tensor(valid_oh).float()

        return fp_tensor, oh_tensor

    def int_to_onehot_single(self, intcode, is_pad):
        intcode = np.array(intcode)
        res = []
        for c in intcode:
            token = self.int2char[c]
            res.append(np.array(self.onehot_dict[token], dtype = np.float32))
        return res

    def int_to_onehot(self, intarray):
        result = []
        for intcode in intarray:
            res = self.int_to_onehot_single(intcode)
            if len(res) > 0:
                result.append(res)

        return result

    def pad_str(self, wordarray):
        nlen = len(wordarray)
        if self.is_pad:
            if nlen > self.max_length:
                raise ValueError('max_length should be larger than length of string when padding is true!')
         
            for i in range(self.max_length - nlen):
                wordarray.append(self.pad_symbol)

        return wordarray, nlen

    def pad_sequences(self, inpx, max_length = None, pad_symbol=' '):
        seqs = [''.join(list(s)) for s in inpx]
       
        if max_length is None:
            max_length = -1
            for seq in seqs:
                max_length = max(max_length, len(seq))

        lengths = []
        for i in range(len(seqs)):
            cur_len = len(seqs[i])
            lengths.append(cur_len)
            seqs[i] = seqs[i] + pad_symbol * (max_length - cur_len)

        return seqs, lengths

    def int_encode_single(self, str, is_pad= False):
        #try:
        wordarray = self.tokenlize(str)
        if is_pad:
            wordarray, nlen = self.pad_str(wordarray)
        else:
            nlen = len(wordarray)
            
        result = np.array([self.char2int[symbol] for symbol in wordarray], dtype = np.float32)
        return result, nlen

    def int_encode(self, seqs, is_pad = False):
        result = []
        lengths = []
             
        stype = type(seqs)
        #if isinstance(seqs, (str)):  #exception:local variable 'str' referenced before assignment
        #   seqs = [seqs]

        for str in seqs:
            inted, nlen  = self.int_encode_single(str, is_pad = is_pad)
            
            if len(inted) > 0:
                result.append(inted)
                lengths.append(nlen)
                  
        result = np.asarray(result)
        return result, lengths
        
    def int_encode_tensor(self, seqs, is_pad= False, device = 'cpu'):
        result,lengths = self.int_encode(seqs, is_pad)

        result = torch.tensor(np.asarray(result), dtype=torch.long, device=device)
        lengths = torch.tensor(np.asarray(lengths), dtype=torch.long, device=device)

        return result, lengths

    def int_decode_single(self, intcode):
        intcode = np.array(intcode)
        token = ''
        for c in intcode:
            token += self.int2char[c]

        return token

    def int_decode(self, intarray):
        result = []
        for intcode in intarray:
            res = self.int_decode_single(intcode)
            if len(res) > 0:
                result.append(res)
                #result.cat(res)
        return result

    def prob_to_int_single(self, prob):
        prob = np.array(prob)
        token = []
        for oh in prob:
            pos = np.argmax(oh)
            token.append(pos)

        return token

    def prob_to_int(self, probarray):
        result = []
        for oh in probarray:
            res = self.prob_to_int_single(oh)
            if len(res) > 0:
                result.append(res)

        return result

    def char2code_tensor(self, inpx, encoder_type, single = False, return_type='long', device = 'cpu',
                         max_length = None):
        inpx = [''.join(list(x)) for x in inpx]

        is_str = False
        if isinstance(inpx, (str)):
            inpx = [inpx]
            is_str = True

        if self.is_pad == False and len(inpx) > 1:
            inpx, inp_seq_len = self.pad_sequences(inpx, max_length = max_length)

        if single:
            if encoder_type == EncoderType.ET_Int:
                inp_tensor, inp_seq_len = self.int_encode_single(inpx, is_pad = self.is_pad)
            elif encoder_type == EncoderType.ET_OH:
                inp_tensor, inp_seq_len = self.int_encode_single(inpx, is_pad = self.is_pad)
            elif encoder_type == EncoderType.ET_Morgan:
                inp_tensor, inp_seq_len = self.int_encode_single(inpx, is_pad = self.is_pad)
            else:
                inp_tensor, inp_seq_len = self.int_encode_single(inpx, is_pad = self.is_pad)
        else:
            if encoder_type == EncoderType.ET_Int:
                inp_tensor, inp_seq_len = self.int_encode(inpx, is_pad = self.is_pad)
            elif encoder_type == EncoderType.ET_OH:
                inp_tensor, inp_seq_len = self.onehot_encode(inpx, is_pad = self.is_pad)
            elif encoder_type == EncoderType.ET_Morgan:
                inp_tensor, inp_seq_len = self.int_encode(inpx, is_pad = self.is_pad)
            else:
                inp_tensor, inp_seq_len = self.int_encode(inpx, is_pad = self.is_pad)
        
        if return_type == 'long':
            inp_tensor = torch.tensor(inp_tensor).long()
        else:
            inp_tensor = torch.tensor(inp_tensor).float()

        inp_tensor = inp_tensor.to(device)
        
        if is_str:
            inp_tensor = inp_tensor[0]

        return inp_tensor
            
#end CTokens    

class STDTokens:
    def __init__(self, *args, **kwargs):
        pass

    def Encoder(self):
        pass

    def Tokens(self):
        pass

class STDTokens_Customed(STDTokens):
    def __init__(self, tokens, *args, **kwargs):
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

        return

    def Encoder(self):
        return self.token_encoder

    def Tokens(self):
        return self.tokens

class  STDTokens_SMI_File(STDTokens):
    def __init__(self, voc_file, add_std = False, *args, **kwargs):
        std_tokens = [' ',  '<', '>', "[",  "]",
                      "-", "=", "#", ":",    #Bonds
                      "(", ")",              #Branches
                      "%",
                      ".",                   #Disconnected Structures
                      '/','\\',              #Configuration Around Double Bonds
                      '*',
                      #'&',
                      ]
        if isinstance(voc_file, str):
            tokens = [x.strip("\r\n ") for x in open(voc_file)]
        elif isinstance(voc_file, (list, np.array, tuple)):
            tokens = []
            for f in voc_file:
                tokens.extend([x.strip("\r\n ") for x in open(f)])

        if add_std:
            tokens = list(set(tokens).union(set(std_tokens)))           
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

        return


    def Encoder(self):
        return self.token_encoder


    def Tokens(self):
        return self.tokens

class STDTokens_Frag_JTVAE_Chembl(STDTokens):
    def __init__(self,  *args, **kwargs):
        self.vocab_path = '../RawData/ChEMBL/chembl_21_1576904.csv_Canonical.smi.jtvae.smi'
        self.tokens = [x.strip("\r\n ") for x in open(self.vocab_path)]
        self.vocab = Vocab(self.tokens )

        self.token_encoder = TokenEncoder.TE_Multiple

        self.tokens_encode = dict((token, i) for i, token in enumerate(self.tokens))
        self.tokens_decode = dict((i, token) for i, token in enumerate(self.tokens))
        return

    def Tokens(self):
        return self.tokens

    def get_index(self, token):
        return self.tokens_encode[token]

    def get_token(self, idx):
        return self.tokens_decode[int(idx)]

class STDTokens_Frag_JTVAE_Zinc(STDTokens):
    def __init__(self,  *args, **kwargs):
        self.vocab_path = '../RawData/JTVAE/data/zinc/JTVAE/all.txt.jtvae.voc'
        self.tokens = [x.strip("\r\n ") for x in open(self.vocab_path)]
        self.vocab = Vocab(self.tokens )

        self.token_encoder = TokenEncoder.TE_Multiple

        self.tokens_encode = dict((token, i) for i, token in enumerate(self.tokens))
        self.tokens_decode = dict((i, token) for i, token in enumerate(self.tokens))
        return

    def Tokens(self):
        return self.tokens

    def get_index(self, token):
        return self.tokens_encode[token]

    def get_token(self, idx):
        return self.tokens_decode[int(idx)]

class STDTokens_Frag_JTVAE_QM9(STDTokens):
    def __init__(self,  *args, **kwargs):
        self.vocab_path = '../RawData/QM/QM9/qm9.smi.jtvae.voc'
        self.tokens = [x.strip("\r\n ") for x in open(self.vocab_path)]
        self.vocab = Vocab(self.tokens )

        self.token_encoder = TokenEncoder.TE_Multiple

        self.tokens_encode = dict((token, i) for i, token in enumerate(self.tokens))
        self.tokens_decode = dict((i, token) for i, token in enumerate(self.tokens))
        return

    def Tokens(self):
        return self.tokens

    def get_index(self, token):
        return self.tokens_encode[token]

    def get_token(self, idx):
        return self.tokens_decode[int(idx)]

class STDTokens_Frag_Brics_Bridge_Zinc(STDTokens):
    #only Brics
    def __init__(self,  *args, **kwargs):
        self.vocab_path = '../RawData/JTVAE/data/zinc/Brics_bridge/all.txt.BRICS_token.voc'
        self.tokens = [x.strip("\r\n ") for x in open(self.vocab_path)]
        self.vocab = Vocab(self.tokens )

        self.token_encoder = TokenEncoder.TE_Multiple

        self.tokens_encode = dict((token, i) for i, token in enumerate(self.tokens))
        self.tokens_decode = dict((i, token) for i, token in enumerate(self.tokens))
        return

    def Tokens(self):
        return self.tokens

    def get_index(self, token):
        return self.tokens_encode[token]

    def get_token(self, idx):
        return self.tokens_decode[int(idx)]

class STDTokens_Frag_Brics_Bridge_Chembl(STDTokens):
    #this is not correct now because it is not processed in desktop
    def __init__(self, *args, **kwargs):
        self.vocab_path = '../RawData/ChEMBL/Brics_bridge/chembl_120.bfs[228]_org.smi.[BRICS_Bridge]_token.voc'
        self.tokens = [x.strip("\r\n ") for x in open(self.vocab_path)]
        self.vocab = Vocab(self.tokens)

        self.token_encoder = TokenEncoder.TE_Multiple

        self.tokens_encode = dict((token, i) for i, token in enumerate(self.tokens))
        self.tokens_decode = dict((i, token) for i, token in enumerate(self.tokens))
        return

    def Tokens(self):
        return self.tokens

    def get_index(self, token):
        return self.tokens_encode[token]

    def get_token(self, idx):
        return self.tokens_decode[int(idx)]
