import rdkit.Chem as Chem

from enum import Enum
class Fragment_Alg(Enum):
    Vanilla      = 0
    JTVAE        = 1
    BRICS_Base   = 2 #only Brics
    BRICS        = 3 #Brics with other breaks, r_link, etc 
    Recap        = 4 #Brics with other breaks
    MMPA         = 5
    Scaffold     = 6
    eMolFrag     = 7


class RDKFragUtil:
    def get_dummy_bond_pair(fraga, fragb):
        #m = Chem.RemoveHs(mol)
        #m = Chem.RWMol(m)

        bond_ids = set()
        nba = None
        nbb = None
        for a in fraga.GetAtoms():
            if a.GetSymbol() == "*":
                #nei_ids = set(na.GetIdx() for na in a.GetNeighbors())
                nei_ids = set(na.GetProp('atomNote') for na in a.GetNeighbors())
                if len(nei_ids) == 1:
                    nba = list(nei_ids)[0]

        for a in fragb.GetAtoms():
            if a.GetSymbol() == "*":
                #nei_ids = set(na.GetIdx() for na in a.GetNeighbors())
                nei_ids = set(na.GetProp('atomNote') for na in a.GetNeighbors())
                if len(nei_ids) == 1:
                    nbb = list(nei_ids)[0]

        return (int(nba),int(nbb))


    def find_break_bonds(mol, frags):
        break_bonds = []

        mol_bonds = mol.GetBonds()

        RDKUtils.add_atom_index(mol)
        RDKUtils.show_mol_with_atommap(mol, atommap = False)

        hierarch = Recap.RecapDecompose(mol)
        leaves = hierarch.GetAllChildren()

        leaf_mols = []
        leaf_mol_atoms = []
        leaf_mol_bonds = []

        for key, value in frags.items():
            leaf = value
            sub_mol = leaf.mol
            leaf_mols.append(sub_mol)
            leaf_mol_bonds.append(sub_mol.GetBonds())
            atoms = sub_mol.GetAtoms()
            leaf_mol_atoms.append(atoms)
            RDKUtils.show_mol_with_atommap(sub_mol, atommap=False)

        for sbonds in leaf_mol_bonds:
            for bond in sbonds:
                #if bond in mol_bonds:

                a1 = bond.GetBeginAtom().GetIdx()
                a2 = bond.GetEndAtom().GetIdx()
                #cliques.append([a1, a2])


        return break_bonds

    def merge_cliques(cliques, single_cliq=[]):
        for c in range(len(cliques) - 1):
            if c >= len(cliques):
                break
            for k in range(c + 1, len(cliques)):
                if k >= len(cliques):
                    break

                share = list(set(cliques[c]) & set(cliques[k]))

                if len(share) > 0 and share not in single_cliq:
                    cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                    cliques[k] = []

            cliques = [c for c in cliques if len(c) > 0]

        cliques = [c for c in cliques if len(c) > 0]

        return cliques

    def bond_in_ring(mol_bonds, startid, endid):
        b = RDKFragUtil.find_bond(mol_bonds, startid, endid)
        if b is not None and b.IsInRing():
            return True
        return False

    def find_bond(mol_bonds, startid, endid):
        for b in mol_bonds:
            if b.GetBeginAtomIdx() == startid and b.GetEndAtomIdx() == endid:
                return b
        return None


    def GetMolFrags(mol_in, break_bonds):
        cutted = Chem.FragmentOnBonds(mol_in, break_bonds, addDummies=False)
        frags = Chem.GetMolFrags(cutted, asMols=True)

        #fg_list = []
        #for fg in frags:
        #    sfg = Chem.MolToSmiles(fg)
        #    fg_list.append(sfg)

        return frags

    def get_break_bond(fmol1, fmol2):
        return
     
    def __get_submol(mol, atom_ids):
        bond_ids = []
        for pair in combinations(atom_ids, 2):
            b = mol.GetBondBetweenAtoms(*pair)

            if b:
                bond_ids.append(b.GetIdx())

        m = Chem.PathToSubmol(mol, bond_ids)
        m.UpdatePropertyCache()
        return m


    def __bonds_to_atoms(mol, bond_ids):
        output = []
        for i in bond_ids:
            b = mol.GetBondWithIdx(i)
            output.append(b.GetBeginAtom().GetIdx())
            output.append(b.GetEndAtom().GetIdx())

        return tuple(set(output))


    def __get_context_env(mol, radius):
        m = Chem.RemoveHs(mol)
        m = Chem.RWMol(m)

        bond_ids = set()
        for a in m.GetAtoms():
            if a.GetSymbol() == "*":
                i = radius
                b = Chem.FindAtomEnvironmentOfRadiusN(m, i, a.GetIdx())
                while not b and i > 0:
                    i -= 1
                    b = Chem.FindAtomEnvironmentOfRadiusN(m, i, a.GetIdx())
                bond_ids.update(b)

        atom_ids = set(RDKFragMMPA.__bonds_to_atoms(m, bond_ids))

        dummy_atoms = []

        for a in m.GetAtoms():
            if a.GetIdx() not in atom_ids:
                nei_ids = set(na.GetIdx() for na in a.GetNeighbors())
                intersect = nei_ids & atom_ids
                if intersect:
                    dummy_atom_bonds = []
                    for ai in intersect:
                        dummy_atom_bonds.append((ai, m.GetBondBetweenAtoms(a.GetIdx(), ai).GetBondType()))
                    dummy_atoms.append(dummy_atom_bonds)

        for data in dummy_atoms:
            dummy_id = m.AddAtom(Chem.Atom(0))
            for atom_id, bond_type in data:
                m.AddBond(dummy_id, atom_id, bond_type)
            atom_ids.add(dummy_id)

        m = RDKFragMMPA.__get_submol(m, atom_ids)

        return m


if __name__ == '__main__':
    i = 0