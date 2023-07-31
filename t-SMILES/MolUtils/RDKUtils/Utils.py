import os
import pandas as pd

import copy

#from MolUtils.RDKUtils.Utils import RDKUtils
class RDKUtils:
    def add_atom_index(mol, prop = 'molAtomMapNumber'):
        n_atoms = mol.GetNumAtoms()
        for i in range( n_atoms ):
            mol.GetAtomWithIdx(i).SetProp(prop, str(mol.GetAtomWithIdx(i).GetIdx()))        

        return mol

    def remove_atommap_info_mol(mol):
        if mol is not None:
            newmol = copy.deepcopy(mol)
            for a in newmol.GetAtoms():
                a.ClearProp('molAtomMapNumber')            
            return newmol
        else:
            return mol
        
    def SaveSmilesWithImage(smiles, filepath, postfix):
        try:
            if isinstance(smiles, (str)):
                smiles = [smiles]
           
            os.makedirs(filepath, exist_ok=True)

            filename = os.path.join(filepath, "gs_" + str(postfix) +".smi")
            RDKUtils.SaveArray2SMI(smiles, filename)    

            #filename = os.path.join(filename +".png")
            #RDKUtils.SMilesToFile(smiles, filename)
        except Exception as e:
            print(e.args)
            pass
        return filename

    def SaveArray2SMI(X, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        df = pd.DataFrame(X)
        df.to_csv(filename, index = False, header=False,na_rep="NULL")
        return
