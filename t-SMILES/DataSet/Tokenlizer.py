import numpy as np
import pandas as pd

from tqdm import tqdm, trange
import string
from Tools.MathUtils import BCMathUtils
import re

#from DataSet.Tokenlizer import Tokenlizer
class Tokenlizer():
    def __init__(self, voc, invalid_token = '&'):
        #this is used by CToken
        self.voc = voc
        self.invalid_token = invalid_token

        self.fdict = {}
        for tk in voc:
            slen = str(len(tk))
            if  slen in self.fdict.keys():
                self.fdict[slen].append(tk)
            else:
                #inpx = [''.join(list(x)) for x in tk]   #split a string to single chars
                self.fdict[slen] = ['']
                self.fdict[slen].append(tk)

        self.fdict = BCMathUtils.dict_sort_key(self.fdict, reverse = True)

        self.max_len = int(list(self.fdict.keys())[0])

        return

    def tokenize(self, inseq):
        if self.max_len == 1:
            tokens = [''.join(list(x)) for x in inseq]
        else:
            pos = 0
            tokens = []
            while pos < len(inseq):
                subs = ['']
                for i in range(1, self.max_len+1,  1):
                    subs.append(inseq[pos : pos + i])
            
                find = False
                for i in range(self.max_len, 0 ,-1):
                    if str(i) not in self.fdict:
                        continue
                    if subs[i] in self.fdict[str(i)]:
                        tokens.append(subs[i])
                        pos += len(subs[i])
                        find = True
                        break       
                    
                if not find:
                    tokens.append(self.invalid_token)
                    pos += 1

        return tokens
           

#------------------------------------------------------------------------------------------------
class TokenlizerUtil:
     def tokenizer_all(sml):   #works well
        pattern = "(\[[^\]]+]|\(|\)|\.|\^|&|=|#|-|:|\+|\\\\|\/|~|@|\?|<|>|\*|\$|\%[0-9]{2}|[0-9])"
        atom_list = ['h', "b", "c", "n", "o", "f",
                    'p', 's',
                    'k', 'v',
                    'y','i',
                    'w','u',
                    "H", "He",
                    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
                    "Na", "Mg","Al", "Si", "P", "S", "Cl", "Ar",
                    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br","Kr",
                    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
                    "Cs", "Ba", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
                    "Fr", "Ra", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",       "Fl",       "Lv",
                    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
                    "Ac", "Th",  "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"
                    ]
        atom_list.remove('Cn')
        atom_list.remove('Sc')
        atom_list.remove('Sn')

        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(sml)]

        atom_list.extend(list(set(tokens))) 

        tokenlizer = Tokenlizer(atom_list, invalid_token = '&')

        tokens = tokenlizer.tokenize(sml)

        assert sml == ''.join(tokens)

        return tokens

     def string_split_single(sml):
        single_tokens = list(sml)
        assert sml == ''.join(single_tokens)
        return single_tokens


def preprocess():       
    smlfile = '../RawData/JTVAE/data/zinc/Brics/all.txt.[BRICS][64]_join.csv'

    token_list = []
    single_token_list = []

    df = pd.read_csv(smlfile, squeeze=True, delimiter=',',header = None) 
    #df.dropna(how="any")
    #smiles_list = [s for s in df.values.astype(str) if s != 'nan']
    smiles_list = list(df.values)
    for i, sml in tqdm(enumerate(smiles_list), total = len(smiles_list),  desc = 'parsing smiles ...'):
        #if len(sml) > maxlen:
        #    continue
        tokens = set(TokenlizerUtil.tokenizer_all(sml))
        token_list.extend(tokens)

        s_tokens = set(TokenlizerUtil.string_split_single(sml))
        single_token_list.extend(s_tokens)

    tokens = list(set(token_list))
    tokens.sort()

    single_token_list = list(set(single_token_list))
    single_token_list.sort()

    output = smlfile + f'.tokens.csv'
    df = pd.DataFrame(tokens)
    df.to_csv(output, index = False, header=False, na_rep="NULL")

    output = smlfile + f'.tokens_single.csv'
    df = pd.DataFrame(single_token_list)
    df.to_csv(output, index = False, header=False, na_rep="NULL")

    return 

if __name__ == '__main__':
    preprocess()

