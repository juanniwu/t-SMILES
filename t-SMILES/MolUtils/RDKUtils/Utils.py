import os
import pandas as pd

import copy

from rdkit import Chem

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

    def getSmarts(mol,atomID, radius, include_H = False):  #JW: include_H is added for reaction, the original code include_H = True
        if radius>0:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol,
                                                    radius = radius, #an integer with the target radius for the environment.
                                                    rootedAtAtom = atomID, #the atom to consider
                                                    #useHs = 0, # (optional) toggles whether or not bonds to Hs that are part of the graph should be included in the results. Defaults to 0.
                                                    #enforceSize = 1, # (optional) If set to False, all bonds within the requested radius is collected. Defaults to 1
                                                    #atomMap: (optional) If provided, it will measure the minimum distance of the atom from the rooted atom (start with 0 from the rooted atom). 
                                                    #The result is a pair of the atom ID and the distance.
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
                                                 atomsToUse,                    #a list of atoms to include in the fragment
                                                 bondsToUse         = env,      #(optional) a list of bonds to include in the fragment if not provided, all bonds between the atoms provided will be included.
                                                 atomSymbols        = symbols,  #(optional) a list with the symbols to use for the atoms in the SMILES. This should have be mol.GetNumAtoms() long.
                                                 allBondsExplicit   = True,     # (optional) if true, all bond orders will be explicitly indicated in the output SMILES. Defaults to false.
                                                 rootedAtAtom       = atomID    #(optional) if non-negative, this forces the SMILES to start at a particular atom. Defaults to -1.
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
        #http://rdkit.chenzhaoqiang.com/FAQs.html
        #sml='[CH3:1][C:2]([N:4]([C:9](=[O:10])[CH3:8])[CH2:5][CH2:6][NH2:7])=[O:3]'
        mol = Chem.MolFromSmiles(sml)
        if mol is None:
            mol = Chem.MolFromSmarts(sml)

        if mol is not None:
            for a in mol.GetAtoms():
                a.ClearProp('molAtomMapNumber')
            # [a.ClearProp('molAtomMapNumber') for a in m.GetAtoms()]

            newsml = Chem.MolToSmiles(mol)    #CC(=O)N(CCN)C(C)=O
        else:
            newsml = ''

        #print(newsml)
        return newsml
