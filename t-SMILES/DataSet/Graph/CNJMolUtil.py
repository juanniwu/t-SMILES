import random
from rdkit import Chem

from Levenshtein import distance as levenshtein   #pip install python-Levenshtein

from DataSet.STDTokens import CTokens

class CNJMolUtil:
    def valid_smiles(sml, ctoken = None, 
                     correct_alg = 'random'
                     #correct_alg = 'best'
                     ):
        if ctoken is None:
            return sml

        osml = 'CC'

        try:
            mol = Chem.MolFromSmiles(sml)

            if mol is None:
                if ctoken is None:
                   osml = 'CC'
                else:
                    if correct_alg == 'random':
                        idx = random.randint(0, ctoken.n_tokens -1 )
                        osml = ctoken.get_token(idx)
                    else:
                        osml =  CNJMolUtil.find_best_match(sml, ctoken) 
             
                print(f'node smile [{sml}] is invalid, which is replaced by [{osml}]')
            else:
                osml = sml
        except Exception as e:
            print('[CNJMolUtil.valid_smiles].exception:', e.args)

        return osml

    def find_best_match(sml, ctoken:CTokens):
        min_dis = float("inf")
        min_pos = 0
        min_token = 'CC'

        try:
            for tok in ctoken.tokens:
                if tok in ctoken.control_tokens():
                    continue

                dis = levenshtein(sml, tok) 
                if dis < min_dis:
                    min_token = tok 
                    min_dis = dis

            if sml.find('*') != -1:
                if min_token.find('*') == -1:
                    min_token = '*' + min_token
        except Exception as e:
            print('[CNJMolUtil.find_best_match].Exception', e.args)
        
        return min_token

    def is_dummy(sml):
        if sml == '&':
            return True
        else:
            return False

    def combine_ex_smiles(bfs_ex_smiles, delimiter = '^'):
        split_char = delimiter
        if bfs_ex_smiles is None or len(bfs_ex_smiles) ==0:
            return '', ''

        sml = bfs_ex_smiles[0]
        skeleton = 'A'
        nlen = len(bfs_ex_smiles)
        for i in range(1, nlen):
            if bfs_ex_smiles[i-1] is not '&' and bfs_ex_smiles[i] is not '&':
                sml += split_char
                skeleton += split_char

            sml = sml + bfs_ex_smiles[i]

            if bfs_ex_smiles[i] is '&':
                skeleton = skeleton + bfs_ex_smiles[i]
            else:
                skeleton = skeleton + 'A'

        return sml, skeleton  
    

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
        return output
