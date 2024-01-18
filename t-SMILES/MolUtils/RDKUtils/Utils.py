import os
import pandas as pd

import copy

from rdkit import Chem

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

        except Exception as e:
            print(e.args)
            pass
        return filename

    def SaveArray2SMI(X, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        df = pd.DataFrame(X)
        df.to_csv(filename, index = False, header=False,na_rep="NULL")
        return

    def getSmarts(mol,atomID, radius, include_H = False):  
        if radius>0:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol,
                                                    radius = radius,
                                                    rootedAtAtom = atomID,
                                                    )
            atomsToUse=[]

            for b in env:
                atomsToUse.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
                atomsToUse.append(mol.GetBondWithIdx(b).GetEndAtomIdx())

            atomsToUse = list(set(atomsToUse))
        else:
            atomsToUse = [atomID]
            env=None

        symbols = []

        for atom in mol.GetAtoms():
            deg = atom.GetDegree()
            isInRing = atom.IsInRing()
            symbol = '[' + atom.GetSmarts()

            if include_H:
                nHs = atom.GetTotalNumHs()
                if nHs:
                    symbol += 'H'
                    if nHs>1:
                        symbol += '%d'%nHs

            if isInRing:
                symbol += ';R'
            else:
                symbol += ';!R'

            symbol += ';D%d'%deg
            symbol += "]"
            symbols.append(symbol)

        try:
            if atomsToUse is not None and len(atomsToUse) > 0:
                smart = Chem.MolFragmentToSmiles(mol,
                                                 atomsToUse,                   
                                                 bondsToUse         = env,      
                                                 atomSymbols        = symbols,  
                                                 allBondsExplicit   = True,    
                                                 rootedAtAtom       = atomID  
                                                 )
            else:
                smart = None

        except (ValueError, RuntimeError) as e:
            print('[RetroTRAE.getSmarts].exception[atom to use error or precondition bond error]:', e.args)
            return None
        return smart


    def atom_isdummy(atom):
        if atom.GetAtomicNum() == 0:
            return True
        else:
            return False

    def remove_dummy_atom(mol):
        edit_mol = Chem.EditableMol(mol)

        ids = []
        if mol is not None:
            for a in mol.GetAtoms():
                if RDKUtils.atom_isdummy(a):
                    ids.append(a.GetIdx())

        for i in sorted(ids, reverse=True):
            edit_mol.RemoveAtom(i)
        
        return edit_mol.GetMol()


    def remove_atommap_info(sml):
        mol = Chem.MolFromSmiles(sml)
        if mol is None:
            mol = Chem.MolFromSmarts(sml)

        if mol is not None:
            for a in mol.GetAtoms():
                a.ClearProp('molAtomMapNumber')

            newsml = Chem.MolToSmiles(mol)    
        else:
            newsml = ''

        return newsml
