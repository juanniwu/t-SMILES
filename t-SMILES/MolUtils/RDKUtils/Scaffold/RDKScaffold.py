from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit import rdBase
rdBase.DisableLog("rdApp.debug")


class RDKScaffold:
    def get_scaffold_frag(sml):
        mol_in = Chem.MolFromSmiles(sml)
        if mol_in is None:
            return set(), None

        if mol_in is not None:
            mol_s = MurckoScaffold.GetScaffoldForMol(mol_in)
            scaffold = Chem.MolToSmiles(mol_s)        

        for atom in mol_in.GetAtoms():
            atom.SetIntProp("atom_idx", atom.GetIdx())
        for bond in mol_in.GetBonds():
            bond.SetIntProp("bond_idx", bond.GetIdx())

        ring_info = mol_in.GetRingInfo()
        bondrings = ring_info.BondRings()
        
        bondring_list = set()
        for br in bondrings:
            bondring_list.update(br)
        bondring_list = sorted(bondring_list)

        all_bonds_idx = [bond.GetIdx() for bond in mol_in.GetBonds()]
        none_ring_bonds_list = []
        for i in all_bonds_idx:
            if i not in bondring_list:
                none_ring_bonds_list.append(i)

        cut_bonds = []  
        for bond_idx in none_ring_bonds_list:
            bgn_atom_idx = mol_in.GetBondWithIdx(bond_idx).GetBeginAtomIdx()
            ebd_atom_idx = mol_in.GetBondWithIdx(bond_idx).GetEndAtomIdx()
            if mol_in.GetBondWithIdx(bond_idx).GetBondTypeAsDouble() == 1.0:
                if mol_in.GetAtomWithIdx(bgn_atom_idx).IsInRing() + mol_in.GetAtomWithIdx(ebd_atom_idx).IsInRing() == 1:
                    t_bond = mol_in.GetBondWithIdx(bond_idx)
                    t_bond_idx = t_bond.GetIntProp("bond_idx")
                    cut_bonds.append(t_bond_idx)

        try:
            if len(cut_bonds) > 0:
                n_cuts = len(cut_bonds)
                dummyLabels = []
                for i in range(n_cuts):
                    dummyLabels.append((0,0))

                cutted = Chem.FragmentOnBonds(mol_in, cut_bonds, addDummies = True, dummyLabels = dummyLabels)
                frags = Chem.GetMolFrags(cutted, asMols=True)

                fg_list = []
                for fg in frags:
                    sfg = Chem.MolToSmiles(fg)
                    fg_list.append(sfg)
            else:
                fg_list = [sml]
        except Exception as e:
            print(f'\r\nException:[{e.args}]:', sml)
            fg_list = []

        return scaffold, fg_list, cut_bonds


def test_sml(): 
    sml = 'Fc1cc(cc(F)c1)C[C@H](NC(=O)c1cc(cc(c1)C)C(=O)N(CCC)CCC)[C@H](O)[C@@H]1[NH2+]CCN(Cc2ccccc2)C1=O'


    scaffold, fg_list, break_bonds = RDKScaffold.get_scaffold_frag(sml)
    print(scaffold)
    print(fg_list)

    sub_mols = []
    sub_mols.append(Chem.MolFromSmiles(sml))
    sub_mols.append(Chem.MolFromSmiles(scaffold))
    sub_mols.extend(Chem.MolFromSmiles(fg) for fg in fg_list)
   

    Draw.MolsToGridImage(sub_mols, molsPerRow = 4, subImgSize=(400, 400)).show()

    return 



if __name__ == '__main__':
    test_sml()
