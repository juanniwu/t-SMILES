import rdkit.Chem as Chem
from rdkit.Chem import BRICS

from MolUtils.RDKUtils.Frag.RDKFragUtil import RDKFragUtil

class RDKFragBrics:

    def decompose_simple(mol):
        return RDKFragBrics.decompose_extra(mol, break_ex = False, break_long_link = False, break_r_bridge = False)

    def decompose_extra(mol,
                        break_ex=True, 
                        break_long_link=True, 
                        break_r_bridge=True,  
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

            brics_bonds = list(BRICS.FindBRICSBonds(mol))
            if len(brics_bonds) == 0:
                return [list(range(n_atoms))], []
            else:
                for bond in brics_bonds:
                    bond = bond[0]
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

                        if break_long_link:  # non-ring and non-ring
                            if not mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(c[0]).IsInRing():
                                breaks.append(c)

                        if break_r_bridge:  # ring-ring bridge
                            if mol.GetAtomWithIdx(c[0]).IsInRing() and mol.GetAtomWithIdx(c[1]).IsInRing():
                                if not RDKFragUtil.bond_in_ring(mol_bonds, c[0], c[1]):
                                    breaks.append(c)

                for b in breaks:
                    if b in cliques:
                        cliques.remove(b)
            # end for break_ex

            cliques = RDKFragUtil.merge_cliques(cliques)

            for b in brics_bonds:
                breaks.append([b[0][0], b[0][1]])

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

            # -----------------------------------------------------------
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
            print('[RDKFragBrics.brics_decomp_extra].Exception: ', Chem.MolToSmiles(mol, kekuleSmiles = True))
            cliques = [list(range(n_atoms))]
            edges = []
            print(e.args)

        return cliques, edges
   

def test():
    sml = r'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'
    mol = Chem.MolFromSmiles(sml)     
    Chem.Kekulize(mol)
    if mol is None:
        print('None')

    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    print(len(brics_bonds))
    return 

if __name__ == '__main__':
    test()

