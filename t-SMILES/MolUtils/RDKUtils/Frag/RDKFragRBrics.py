
import copy

import rdkit.Chem as Chem

from MolUtils.RDKUtils.Utils import RDKUtils
from MolUtils.RDKUtils.Frag.RDKFragUtil import RDKFragUtil

from MolUtils.rBRICS.rBRICS_public import FindrBRICSBonds

class RDKFragRBrics:
    def decompose_dummy(mol,
                    break_ex        = False, 
                    break_long_link = False, 
                    break_r_bridge  = False,  
                    ):
                
        try:
            sml = Chem.MolToSmiles(mol, kekuleSmiles = True)
            mol = Chem.MolFromSmiles(sml)

            Chem.Kekulize(mol)   # gets different cut bonds, it's a bug i think

        except Exception as e:
            print('[RDKFragMMPA.decompose_dummy].Exception-ignore:', e.args)


        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            cliques = [set(list(range(n_atoms)))]  
            motif_str = [Chem.MolToSmiles(mol, kekuleSmiles = True)]
            edges = []
            dummy_atoms = []
            frags_smarts = motif_str
            return cliques, edges, motif_str, dummy_atoms, frags_smarts

        cliques = []
        breaks = []
        edges = []

        atom_cliques = {}
        for i in range(n_atoms):
            atom_cliques[i] = set()  

        try:
            mol_bonds = mol.GetBonds()
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom().GetIdx()
                a2 = bond.GetEndAtom().GetIdx()
                cliques.append([a1, a2])

            single_cliq = []

            break_bonds, bond_idxs = RDKFragRBrics.find_break_bonds(mol)
            if len(break_bonds) > 0:
                frags_mol, frags_smarts, frags_sml, motifs_aidx, dummy_atoms = RDKFragUtil.GetMolFrags(mol, bond_idxs, addDummies = True)

                cliques = motifs_aidx
                motif_str = frags_sml

                for cbond in break_bonds:
                    bs = cbond[0]
                    be = cbond[1]
                    es = 0
                    ee = 0
                    for i, c in enumerate(cliques):
                        if bs in c:
                            es = i
                        if be in c:
                            ee = i
                    edges.append((es, ee))
            else:
                cliques = [set(list(range(n_atoms)))]  
                motif_str = [Chem.MolToSmiles(mol, kekuleSmiles = True)]
                edges = []
                dummy_atoms = []
                frags_smarts = motif_str

            edges = list(set(edges))
        
        except Exception as e:
            print(e.args)
            print('RDKFragRBrics Exception: ', Chem.MolToSmiles(mol))
            cliques = [list(range(n_atoms))]
            edges = []
            dummy_atoms = []
            motif_str = [Chem.MolToSmiles(mol, kekuleSmiles = True )]
            frags_smarts = motif_str

        return cliques, edges, motif_str, dummy_atoms, frags_smarts


    def find_break_bonds(mol_in, maxCuts = 1):
        mol = copy.deepcopy(mol_in)
        RDKUtils.add_atom_index(mol, prop = 'atomNote')

        break_bonds = []

        mol_bonds = mol.GetBonds()
        n_bonds = len(mol_bonds)

        cut_bonds = FindrBRICSBonds(mol)        

        bond_idxs = []
        for bd in cut_bonds:
            bd = bd[0]
            bond = mol.GetBondBetweenAtoms(bd[0], bd[1])
            bidx = bond.GetIdx() #[1, 2, 3, 4, 6]
            bond_idxs.append(bidx)
            break_bonds.append(bd)

        return break_bonds, bond_idxs


def test_id():
    sml = 'FC(F)(F)C1=NS(=O)(=O)C2=C(N1)C=C(C=C2)C1CCCC1'  #ring breaken,

    mol = Chem.MolFromSmiles(sml)
    #Chem.Kekulize(mol)


    cliques, edges, motif_str, dummy_atoms, motif_smarts = RDKFragRBrics.decompose_dummy(mol)
    print('[motif_str]:', motif_str)
    print('[motif_smarts]:', motif_smarts)

    return 


if __name__ == '__main__':

    test_id()