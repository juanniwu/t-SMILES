import random
import copy

import rdkit
from rdkit import Chem

#from DataSet.Graph.CNJMolUtil import CNJMolUtil
class CNJMolUtil:
    def valid_smiles(sml, ctoken = None):
        mol = Chem.MolFromSmiles(sml)
        if mol is None:
            if ctoken is None:
               osml = 'CC'
            else:
                idx = random.randint(0, ctoken.n_tokens -1 )
                osml = ctoken.get_token(idx)
            print(f'node smile [{sml}] is invalid, which is replaced by [{osml}]')
        else:
            osml = sml
        return osml

    def is_dummy(sml):
        if sml == '&':
            return True
        else:
            return False

    def combine_ex_smiles(bfs_ex_smiles, delimiter = '^'):
        split = delimiter
        if bfs_ex_smiles is None or len(bfs_ex_smiles) ==0:
            return ''

        sml = bfs_ex_smiles[0]
        nlen = len(bfs_ex_smiles)
        for i in range(1, nlen):
            if bfs_ex_smiles[i-1] is not '&' and bfs_ex_smiles[i] is not '&':
                sml += split
            sml = sml + bfs_ex_smiles[i]

        return sml

    def split_ex_smiles(ex_smiles, delimiter = '^'):
        split = delimiter

        output = []
        words = ex_smiles.split(split)
        for w in words:
            group = ''
            for s in w:
                if s == '&':
                    if group != '':
                        output.append(group)  
                        group = ''
                    output.append('&')  
                else:
                    group +=s
            if group != '':
                output.append(group) 
        #words = [w.split('&') for w in words]
        return output
