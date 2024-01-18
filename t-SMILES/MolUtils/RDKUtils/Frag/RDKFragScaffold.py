import copy

import rdkit.Chem as Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from MolUtils.RDKUtils.Utils import RDKUtils
from MolUtils.RDKUtils.Frag.RDKFragUtil import RDKFragUtil

class RDKFragScaffold:
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

            break_bonds, bond_idxs = RDKFragScaffold.find_break_bonds(mol)
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

            break_bonds, bond_idxs = RDKFragScaffold.find_break_bonds(mol)

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
            # end for break_ex

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
            print('RDKFragMMPA Exception: ', Chem.MolToSmiles(mol, kekuleSmiles = True))
            cliques = [list(range(n_atoms))]
            edges = []

        return cliques, edges

   
    def find_break_bonds(mol_in):
        mol = copy.deepcopy(mol_in)
        RDKUtils.add_atom_index(mol, prop = 'atomNote')
           
        break_bonds = []

        mol_atoms =  [int(atom.GetProp('atomNote')) for atom in  mol.GetAtoms()]
        mol_bonds = mol.GetBonds()
        n_bonds = len(mol_bonds)
    
        mol_sf = MurckoScaffold.GetScaffoldForMol(mol)
        sml_sf = Chem.MolToSmiles(mol_sf)        
        sf_bonds = mol_sf.GetBonds()
        sf_atoms = [int(atom.GetProp('atomNote')) for atom in  mol_sf.GetAtoms()]        

        for bond in mol_bonds:
            if bond not in sf_bonds:
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()

                nba = int(a1.GetProp('atomNote'))
                nbb = int(a2.GetProp('atomNote'))                   

                if ( nba in sf_atoms and nbb not in sf_atoms) or (nba  not in sf_atoms and nbb in sf_atoms):
                    break_bonds.append((nba, nbb) if nba <nbb else (nbb, nba ))

        bond_idxs = []
        for bd in break_bonds:
            bond = mol.GetBondBetweenAtoms(bd[0], bd[1])
            bidx = bond.GetIdx() #[1, 2, 3, 4, 6]
            bond_idxs.append(bidx)

        return break_bonds, bond_idxs


    def get_frag(mol_in):
        break_bonds , bond_idxs= RDKFragScaffold.find_break_bonds(mol_in)   
        frags_mol, frags_smarts, frags_sml, motifs_aidx, dummy_atoms = RDKFragUtil.GetMolFrags(mol_in, break_bonds)

        mol_list = [mol_in]
        mol_list.extend(frags_mol)

        return break_bonds, frags_mol


def test_sml():
    sml = 'Fc1cc(cc(F)c1)C[C@H](NC(=O)c1cc(cc(c1)C)C(=O)N(CCC)CCC)[C@H](O)[C@@H]1[NH2+]CCN(Cc2ccccc2)C1=O'
    mol = Chem.MolFromSmiles(sml)

    RDKFragScaffold.get_frag(mol)

    return 



if __name__ == '__main__':
    test_sml()
