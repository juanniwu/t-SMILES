import copy

import rdkit.Chem as Chem
from rdkit.Chem import BRICS

from MolUtils.RDKUtils.Frag.RDKFragUtil import RDKFragUtil
from MolUtils.RDKUtils.Utils import RDKUtils

class RDKFragBricsDummy:
    def decompose_dummy(mol,
                    break_ex        = False,  
                    break_long_link = False,
                    break_r_bridge  = False, 
                    ):
                
        try:
            sml = Chem.MolToSmiles(mol, kekuleSmiles = True)
            mol = Chem.MolFromSmiles(sml)

            Chem.Kekulize(mol)   #gets different cut bonds, it's a bug i think

        except Exception as e:
            print('[RDKFragBricsDummy.decompose_dummy].Exception-ignore:', e.args)

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

            break_bonds, bond_idxs = RDKFragBricsDummy.find_break_bonds(mol)
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
                frags_smarts= motif_str
        
        except Exception as e:
            print(e.args)
            print('RDKFragMMPA Exception: ', Chem.MolToSmiles(mol))
            cliques = [list(range(n_atoms))]
            edges = []
            motif_str = [Chem.MolToSmiles(mol, kekuleSmiles = True)]
            frags_smarts = motif_str

        return cliques, edges, motif_str, dummy_atoms, frags_smarts

    def decompose(mol,
                break_ex        = False, 
                break_long_link = False, 
                break_r_bridge  = False, 
                ):
        cliques = None    
        edges   = None     
        motif_str = None
        dummy_atom = None
        try:
            Chem.Kekulize(mol)
            sml = Chem.MolToSmiles(mol, kekuleSmiles = True )
            smarts = Chem.MolToSmarts(mol)

            moltree = CBricsTree(sml,
                             min_length = 3, 
                             addDummies = True, 
                             extra_cut = False, #False,True
                             atommap = False,
                             )

            root = moltree.root
            cut_bonds = moltree.get_cutbonds()
            motif_str, cliques, motif_smarts = moltree.get_motif()
            dummy_atom = root.dummy_atoms
            n_atoms = root.n_atoms
            atom_env = root.atom_env      
            
            #----------------
            edges = []
            for cbond in cut_bonds:
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

        except Exception as e:
            print('[RDKFragBricsDummy.brics_decomp_extra].Exception: ', Chem.MolToSmiles(mol))
            cliques = [list(range(n_atoms))]
            edges = []
            motif_str = [Chem.MolToSmiles(mol)]
            dummy_atom = []
            atom_env = []
            cut_bonds = []
            motif_smarts = motif_str
            print(e.args)

        return cliques, edges, motif_str, dummy_atom, atom_env, cut_bonds, motif_smarts


    def find_break_bonds(mol_in, maxCuts = 1):
        mol = copy.deepcopy(mol_in)
        RDKUtils.add_atom_index(mol, prop = 'atomNote')

        break_bonds = []

        mol_bonds = mol.GetBonds()
        n_bonds = len(mol_bonds)

        cut_bonds = list(BRICS.FindBRICSBonds(mol))
        
 
        bond_idxs = []
        for bd in cut_bonds:
            bd = bd[0]
            bond = mol.GetBondBetweenAtoms(bd[0], bd[1])
            bidx = bond.GetIdx() #[1, 2, 3, 4, 6]
            bond_idxs.append(bidx)
            break_bonds.append(bd)

        return break_bonds, bond_idxs


def test():
    sml = r'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F' #Celecoxib

    mols = []
    mmm = Chem.MolFromSmiles(sml)
    mols.append(mmm)

    mols.append(Chem.MolFromSmiles('P'))

    maxCuts = 2

    fr = rdMMPA.FragmentMol(mmm, maxCuts = maxCuts, resultsAsMols = True, maxCutBonds = 50)
    
    for f in fr:
        ff = Chem.GetMolFrags(f[1], asMols=True)
        mols.append(ff[0])
        mols.append(ff[1])

    mols.append(Chem.MolFromSmiles('P'))
    mols.append(Chem.MolFromSmiles('P'))

    Draw.MolsToGridImage(mols, molsPerRow = 8,).show()

    return 

if __name__ == '__main__':
    test()

