import copy

import rdkit.Chem as Chem

from MolUtils.RDKUtils.Utils import RDKUtils
from MolUtils.RDKUtils.Frag.RDKFragUtil import RDKFragUtil

from rdkit.Chem import Draw
from rdkit.Chem import rdMMPA

class RDKFragMMPA:
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

            break_bonds, bond_idxs = RDKFragMMPA.find_break_bonds(mol)
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
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]], []

        cliques = []
        breaks = []

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

            break_bonds, bond_idxs = RDKFragMMPA.find_break_bonds(mol)

            if len(break_bonds) == 0:
                return [list(range(n_atoms))], []
            else:
                for bond in break_bonds:
                    if [bond[0], bond[1]] in cliques:
                        cliques.remove([bond[0], bond[1]])
                    else:
                        cliques.remove([bond[1], bond[0]])

            if break_ex:
                for c in cliques:
                    if len(c) > 1:
                        if mol.GetAtomWithIdx(c[0]).IsInRing() and not mol.GetAtomWithIdx(c[1]).IsInRing():
                            breaks.append(c)

                        if mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(c[0]).IsInRing():
                            breaks.append(c)

                        if break_long_link:  
                            if not mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(c[0]).IsInRing():
                                breaks.append(c)

                        if break_r_bridge: 
                            if mol.GetAtomWithIdx(c[0]).IsInRing() and mol.GetAtomWithIdx(c[1]).IsInRing():
                                if not RDKFragUtil.bond_in_ring(mol_bonds, c[0], c[1]):
                                    breaks.append(c)

                for b in breaks:
                    if b in cliques:
                        cliques.remove(b)

            cliques = RDKFragUtil.merge_cliques(cliques)

            for b in break_bonds:
                breaks.append([b[0], b[1]])

            # --------------------
            break_atom_bonds = {}
            for i, b in enumerate(breaks):
                b0 = b[0]
                b1 = b[1]
                if b0 in break_atom_bonds:
                    break_atom_bonds[b0].append(b)
                else:
                    break_atom_bonds[b0] = [b]

                if b1 in break_atom_bonds:
                    break_atom_bonds[b1].append(b)
                else:
                    break_atom_bonds[b1] = [b]

            # ---------------------
            single_cliq = []

            for key, value in break_atom_bonds.items():
                aid = key
                atom = mol.GetAtomWithIdx(aid)

                if len(value) > 2:
                    cliques.append([key])
                    single_cliq.append([key])
                elif len(value) == 2 and mol.GetAtomWithIdx(key).IsInRing():
                    cliques.append([key])
                    single_cliq.append([key])

                for i in range(len(value)):
                    b = value[i]
                    if [b[0], b[1]] not in cliques and [b[1], b[0]] not in cliques:
                        cliques.append(b)

            for i in range(n_atoms):
                atom_cliques[i] = set()

            for i, cliq in enumerate(cliques):
                for a in cliq:
                    atom_cliques[a].add(i)

            for key, value in atom_cliques.items():
                if len(value) >= 3:
                    if [key] not in cliques:
                        cliques.append([key])
                        single_cliq.append([key])

            # --------------------
            # edges
            edges = []
            singles = set()
            for s in range(len(cliques)):
                s_cliq = cliques[s]
                if len(s_cliq) == 1:
                    singles.add(s)
                    continue
                for e in range(s + 1, len(cliques)):
                    e_cliq = cliques[e]
                    if len(e_cliq) == 1:
                        singles.add(e)
                        continue
                    share = list(set(s_cliq) & set(e_cliq))
                    if len(share) > 0 and share not in single_cliq:
                        edges.append((s, e))

            for i in singles:
                s_cliq = cliques[i]
                for cid in range(len(cliques)):
                    if i == cid:
                        continue
                    share = list(set(cliques[i]) & set(cliques[cid]))
                    if len(share) > 0:
                        if i < cid:
                            edges.append((i, cid))
                        else:
                            edges.append((cid, i))

        except Exception as e:
            print(e.args)
            print('RDKFragMMPA Exception: ', Chem.MolToSmiles(mol, kekuleSmiles = True ))
            cliques = [list(range(n_atoms))]
            edges = []

        return cliques, edges

   
    def find_break_bonds(mol_in, maxCuts = 1):
        mol = copy.deepcopy(mol_in)
        RDKUtils.add_atom_index(mol, prop = 'atomNote')

        break_bonds = []

        mol_bonds = mol.GetBonds()
        n_bonds = len(mol_bonds)

        sub_mols = []

        sub_mols.append(mol)
        sub_mols.append(Chem.MolFromSmiles('C'))

        fr = rdMMPA.FragmentMol(mol, maxCuts = maxCuts, resultsAsMols = True, maxCutBonds = n_bonds)
        for f in fr:
            ff = Chem.GetMolFrags(f[1], asMols=True)
            sub_mols.append(ff[0])  
            sub_mols.append(ff[1])  

            (nba, nbb) = RDKFragUtil.get_dummy_bond_pair(ff[0], ff[1])
            break_bonds.append((nba, nbb))

        bond_idxs = []
        for bd in break_bonds:
            bond = mol.GetBondBetweenAtoms(bd[0], bd[1])
            bidx = bond.GetIdx() 
            bond_idxs.append(bidx)


        return break_bonds, bond_idxs



def test():
    sml = 'CC(=O)Nc1c2C(=O)N(C3CCCCC3)[C@@](C)(C(=O)NC3CCCCC3)Cn2c2ccccc12'

    mols = []
    mmm = Chem.MolFromSmiles(sml)
    mols.append(mmm)

    mols.append(Chem.MolFromSmiles('P'))

    maxCuts = 2

    fr = rdMMPA.FragmentMol(mmm, maxCuts = maxCuts, resultsAsMols = True, maxCutBonds = 50)
    fr_sml = rdMMPA.FragmentMol(mmm, maxCuts = maxCuts, resultsAsMols = False, maxCutBonds = 50)
    fr_sml = set(fr_sml)
    
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

   