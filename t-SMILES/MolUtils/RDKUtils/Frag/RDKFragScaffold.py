import copy

import rdkit.Chem as Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from MolUtils.RDKUtils.Utils import RDKUtils
from MolUtils.RDKUtils.Frag.RDKFragUtil import RDKFragUtil

class RDKFragScaffold:
    def decompose(mol,
                break_ex        = False,  # do ex-action besides basic BRICS algorithm
                break_long_link = False,  # non-ring and non-ring
                break_r_bridge  = False,  # ring-ring bridge
                ):
        #RDKUtils.show_mol_with_atommap(mol, atommap = True)

        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]], []

        cliques = []
        breaks = []

        atom_cliques = {}
        for i in range(n_atoms):
            atom_cliques[i] = set()  # atom-cliques map

        try:
            mol_bonds = mol.GetBonds()
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom().GetIdx()
                a2 = bond.GetEndAtom().GetIdx()
                cliques.append([a1, a2])

            single_cliq = []

            break_bonds = RDKFragScaffold.find_break_bonds(mol)

            if len(break_bonds) == 0:
                return [list(range(n_atoms))], []
            else:
                for bond in break_bonds:
                    #bond = bond[0]  #This is not need here, but need for RDK bond
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

            for b in break_bonds:
                #b = b[0]  #This is not need here, but need for RDK bond
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
                    # the shared point as a center clique
                    cliques.append([key])
                    single_cliq.append([key])
                elif len(value) == 2 and mol.GetAtomWithIdx(key).IsInRing():
                    cliques.append([key])
                    single_cliq.append([key])

                # if len(value) == 1 or len(value) == 2:
                for i in range(len(value)):
                    b = value[i]
                    if [b[0], b[1]] not in cliques and [b[1], b[0]] not in cliques:
                        cliques.append(b)

            # -------------------
            # find exteral single_cliq when it is created by breaks and no breaks
            # could be tested using BRICS_Base algorithm
            # smls = 'CC(=O)Nc1c2C(=O)N(C3CCCCC3)[C@@](C)(C(=O)NC3CCCCC3)Cn2c2ccccc12'

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
            print('RDKFragMMPA Exception: ', Chem.MolToSmiles(mol))
            cliques = [list(range(n_atoms))]
            edges = []

        return cliques, edges

   
    #def find_break_bonds(mol_in):
    #    mol = copy.deepcopy(mol_in)
    #    sml = Chem.MolToSmiles(mol_in)
    #    break_bond_list = []

    #    #scaffold, fg_list, break_bonds = RDKFragScaffold.get_scaffold_frag(sml)   
    #    break_bonds = RDKFragScaffold.get_scaffold_frag(mol_in)   
        
    #    sub_mols = []
    #    sub_mols.append(mol_in)
    #    sub_mols.append(Chem.MolFromSmiles(scaffold))
    #    sub_mols.extend(Chem.MolFromSmiles(fg) for fg in fg_list)

    #    Draw.MolsToGridImage(sub_mols, molsPerRow = 4, subImgSize=(400, 400)).show()

    #    mol_bonds = mol.GetBonds()
    #    for bond in mol.GetBonds():
    #        bidx =  bond.GetIdx()
    #        if bidx in break_bonds:
    #            a1 = bond.GetBeginAtom().GetIdx()
    #            a2 = bond.GetEndAtom().GetIdx()
    #            break_bond_list.append((a1, a2))

    #    return break_bond_list

    def find_break_bonds(mol_in):
        mol = copy.deepcopy(mol_in)
        RDKUtils.add_atom_index(mol, prop = 'atomNote')
        #RDKUtils.show_mol_with_atommap(mol, atommap = True)    
           
        break_bonds = []

        mol_atoms =  [int(atom.GetProp('atomNote')) for atom in  mol.GetAtoms()]
        mol_bonds = mol.GetBonds()
        n_bonds = len(mol_bonds)
    
        mol_sf = MurckoScaffold.GetScaffoldForMol(mol)
        sml_sf = Chem.MolToSmiles(mol_sf)        
        sf_bonds = mol_sf.GetBonds()
        sf_atoms = [int(atom.GetProp('atomNote')) for atom in  mol_sf.GetAtoms()]
        

        #Draw.MolsToGridImage([mol_in, mol_sf], molsPerRow = 4, subImgSize=(400, 400)).show()

        #fw = MurckoScaffold.MakeScaffoldGeneric(mol_sf)
        #http://www.rdkit.org/docs/GettingStartedInPython.html
        #Draw.MolsToGridImage([mol_in, mol_sf, fw], molsPerRow = 4, subImgSize=(400, 400)).show()

        for bond in mol_bonds:
            if bond not in sf_bonds:
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()

                nba = int(a1.GetProp('atomNote'))
                nbb = int(a2.GetProp('atomNote'))                   

                if ( nba in sf_atoms and nbb not in sf_atoms) or (nba  not in sf_atoms and nbb in sf_atoms):
                    #break_bonds.append(bond.GetIdx())
                    break_bonds.append((nba, nbb) if nba <nbb else (nbb, nba ))


        return break_bonds


    def get_frag(mol_in):
        break_bonds = RDKFragScaffold.find_break_bonds(mol_in)   
        frags = RDKFragUtil.GetMolFrags(mol_in, break_bonds)

        mol_list = [mol_in]
        mol_list.extend(frags)

        #Draw.MolsToGridImage(mol_list, molsPerRow = 4, subImgSize=(400, 400)).show()

        return break_bonds, frags


def test_sml():
    #sml = 'CC(C)(C)C1=CC=C(C=C1)C(=O)CC(=O)C2=CC=C(C=C2)OC'   
    #sml = 'C[NH+](C/C=C/c1ccco1)CCC(F)(F)F'  
    #sml = 'CC1OC(OCC=C2CCC3C4CCC5Cc6nc7c(nc6CC5(C)C4C(O)CC23C)CC2CCC3C4CCC(=CCOC5OC(C)C(O)C(O)C5O)C4(C)CC(O)C3C2(C)C7)C(O)C(O)C1O'
    sml = 'Fc1cc(cc(F)c1)C[C@H](NC(=O)c1cc(cc(c1)C)C(=O)N(CCC)CCC)[C@H](O)[C@@H]1[NH2+]CCN(Cc2ccccc2)C1=O'
    mol = Chem.MolFromSmiles(sml)

    #break_bonds = RDKFragScaffold.find_break_bonds(mol)

    RDKFragScaffold.get_frag(mol)


    return 



if __name__ == '__main__':
    test_sml()
