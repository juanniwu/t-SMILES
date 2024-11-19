

import numpy as np
import pandas as pd

from tqdm import tqdm, trange

import rdkit
from rdkit import Chem


from MolUtils.RDKUtils.Frag.RDKFragUtil import RDKFragUtil

class RDKEnumerateSmiles:
    def enumerate_smiles(base_sml, n_samples, code = 'smiles', isomericSmiles = True, kekuleSmiles = False):  #default values are same as RDKit
        sml_list = []
        s = RDKEnumerateSmiles.randomize_smiles_simple(base_sml, code, canonical = True, isomericSmiles = isomericSmiles, kekuleSmiles = kekuleSmiles)    
        sml_list.append(s)

        for i in range(n_samples - 1):
            s = RDKEnumerateSmiles.randomize_smiles_simple(base_sml, code, canonical = False, isomericSmiles = isomericSmiles, kekuleSmiles = kekuleSmiles)    
            sml_list.append(s)

        return sml_list

    def randomize_smiles_simple(smiles, code = 'smiles', canonical = False, isomericSmiles = False,  kekuleSmiles = False):
        """Perform a randomization of a SMILES string must be RDKit sanitizable"""
        if code == 'smiles':
            mol = Chem.MolFromSmiles(smiles)
        else:
            mol = Chem.MolFromSmarts(smiles)

        kekule_error = False
        if mol is not None:
            ans = list(range(mol.GetNumAtoms()))
            np.random.shuffle(ans)
             
            if code == 'smiles': 
                try:
                    Chem.Kekulize(mol)
                except Exception as e:
                    #print('[RDKEnumerator.randomize_smiles_simple].Exception1:', e.args)
                    #print('[RDKEnumerator.randomize_smiles_simple].Exception1:', smiles)
                    kekule_error = True
                    #return smiles

            #even canonical is set as True, get different smiles with different rootedAtAtom
            if canonical:
                rootedAtAtom = -1
            else:
                rootedAtAtom = ans[0]

            
            if code == 'smiles':
                try:
                    #Chem.SanitizeMol(mol)      # could not call SanitizeMol               
                    sml = Chem.MolToSmiles(mol,
                                        rootedAtAtom = rootedAtAtom,
                                        #canonical = canonical,
                                        isomericSmiles = isomericSmiles,
                                        kekuleSmiles = True)
                except Exception as e:
                    print('[RDKEnumerator.randomize_smiles_simple].Exception2:', e.args)
                    print('[RDKEnumerator.randomize_smiles_simple].Exception2:', smiles)            
                    sml = Chem.MolToSmiles(mol, kekuleSmiles = True)

            else: #for smarts, isomericSmiles could not be set if id of[3*] need to be kept:'[3*]CC1=C(C)C2=C(S1)C(=O)C(C([1*])=O)=CN2C'
                sml = Chem.MolToSmiles(mol,
                                        rootedAtAtom = rootedAtAtom, 
                                        #canonical = canonical,
                                        kekuleSmiles = True
                                        )
                if not RDKFragUtil.cut_is_valid([sml]):
                    sml = smiles
        else:
            sml = smiles


        #sml = '[3*]CC1=C(C)C2=C(S1)C(=O)C(C([1*])=O)=CN2C'  
        #mol = Chem.MolFromSmiles(sml)
        #Chem.Kekulize(mol)
        #1: print(Chem.MolToSmiles(mol,canonical = True, rootedAtAtom = 3, kekuleSmiles = True))   #1 and 2 get different strings
        #2: print(Chem.MolToSmiles(mol,canonical = True, rootedAtAtom = 5, kekuleSmiles = True))
        #3: print(Chem.MolToSmiles(mol,canonical = False, rootedAtAtom = 3, kekuleSmiles = True))
        #4: print(Chem.MolToSmiles(mol,canonical = False, rootedAtAtom = 5, kekuleSmiles = True))

        return sml
