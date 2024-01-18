from rdkit import Chem
from rdkit import rdBase
rdBase.DisableLog("rdApp.info")

#from MolUtils.RDKUtils.RDKEdit import RDKEdit

class RDKEdit():
    def copy_atom(atom, fromcharge = False, atommap=False):
        new_atom = Chem.Atom(atom.GetSymbol())
        if fromcharge:
            new_atom.SetFormalCharge(atom.GetFormalCharge())
        if atommap: 
            new_atom.SetAtomMapNum(atom.GetAtomMapNum())
        return new_atom

    def rdk_rw_mol(mol):
        mw = Chem.RWMol(mol)
        mw.ReplaceAtom(4,Chem.Atom(7))
        mw.AddAtom(Chem.Atom(6))
        mw.AddAtom(Chem.Atom(6))
        mw.AddBond(6,7,Chem.BondType.SINGLE)
        mw.AddBond(7,8,Chem.BondType.DOUBLE)
        mw.AddBond(8,3,Chem.BondType.SINGLE)
        mw.RemoveAtom(0)
        mw.GetNumAtoms()
        return mol

    def edit_single_atom_mol(mol):
        new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
        for atom in mol.GetAtoms():
            new_atom = RDKEdit.copy_atom(atom)

            try:
                atomnote = atom.GetProp('atomNote')
                new_atom.SetProp('atomNote', atomnote)
            except Exception as e:
                print('edit_single_atom_mol', e.args)

            new_mol.AddAtom(new_atom)
            mol = new_mol.GetMol()
            mol = Chem.RemoveHs(mol)
        return mol