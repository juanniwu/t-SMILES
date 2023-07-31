import copy

import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdMMPA

from MolUtils.RDKUtils.Utils import RDKUtils
from MolUtils.RDKUtils.Frag.RDKFragUtil import RDKFragUtil

class RDKFragMMPA:
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

            break_bonds = RDKFragMMPA.find_break_bonds(mol)

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

   
    def find_break_bonds(mol_in, maxCuts = 1):
        mol = copy.deepcopy(mol_in)
        RDKUtils.add_atom_index(mol, prop = 'atomNote')
        #RDKUtils.show_mol_with_atommap(mol, atommap = False)          

        break_bonds = []

        mol_bonds = mol.GetBonds()
        n_bonds = len(mol_bonds)

        #RDKUtils.show_mol_with_label(mol, prop = 'atomNote')          

        sub_mols = []

        sub_mols.append(mol)
        sub_mols.append(Chem.MolFromSmiles('C'))

        fr = rdMMPA.FragmentMol(mol, maxCuts = maxCuts, resultsAsMols = True, maxCutBonds = n_bonds)
        for f in fr:
            ff = Chem.GetMolFrags(f[1], asMols=True)
            sub_mols.append(ff[0])  
            sub_mols.append(ff[1])  
            #print(Chem.MolToSmiles(ff[0]))
            #print(Chem.MolToSmiles(ff[1]))

            (nba, nbb) = RDKFragUtil.get_dummy_bond_pair(ff[0], ff[1])
            break_bonds.append((nba, nbb))

            #RDKUtils.show_mol_with_atommap([ff[0], ff[1]], atommap = False)          
            #n_mol = RDKFragMMPA.__get_context_env(ff[1], radius = 1)
            #RDKUtils.show_mol_with_atommap([n_mol], atommap = False)          

            #smol1 = RDKUtils.remove_dummy_atom(ff[0])
            #sub_mols.append(smol1)

            #smol2 =  RDKUtils.remove_dummy_atom(ff[1])
            #sub_mols.append(smol2)

            #RDKUtils.show_mol_with_atommap([smol1, smol2], atommap = False)          

        #Draw.MolsToGridImage(sub_mols, molsPerRow = 4, subImgSize=(400, 400)).show()


        return break_bonds

def test_sml():
    #sml = 'C[NH+](C/C=C/c1ccco1)CCC(F)(F)F'  
    sml = 'CC1OC(OCC=C2CCC3C4CCC5Cc6nc7c(nc6CC5(C)C4C(O)CC23C)CC2CCC3C4CCC(=CCOC5OC(C)C(O)C(O)C5O)C4(C)CC(O)C3C2(C)C7)C(O)C(O)C1O'
    mol = Chem.MolFromSmiles(sml)

    break_bonds = RDKFragMMPA.find_break_bonds(mol, maxCuts = 1)

    return 

def test():
    #https://zhuanlan.zhihu.com/p/389763022
    #http://rdkit.org/docs/source/rdkit.Chem.rdMMPA.html?highlight=rdmmpa
    #http://rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html?highlight=atompairsparameters#rdkit.Chem.rdMolDescriptors.AtomPairsParameters
    #FragmentMol( (Mol)mol [ , (int)maxCuts=3 [ , (int)maxCutBonds=20 [ , (str)pattern='[#6+0;!$(*= , #[!#6])]! @!=!#[*]' [ , (bool)resultsAsMols=True ] ] ] ] )
    #FragmentMol( (Mol)mol, (int)minCuts, (int)maxCuts, (int)maxCutBonds [, (str)pattern='[#6+0;!$( =,#[!#6])]!@ !=!#[ ]' [, (bool)resultsAsMols=True]])
    #FragmentMol( (Mol)mol, (AtomPairsParameters)bondsToCut [, (int)minCuts=1 [, (int)maxCuts=3 [, (bool)resultsAsMols=True]]]) 
    
    #sml = 'CC(=O)Nc1c2C(=O)N(C3CCCCC3)[C@@](C)(C(=O)NC3CCCCC3)Cn2c2ccccc12'
    #sml = 'C(C(F)F)(C(C1CCCCC1)CC1CCCC1)CN1C2SC(Cl)=CC=2S(=O)(=O)NC1'
    #sml = 'NC(=O)C1CCC(CNc2cc(-c3ccccc3)nc3ccnn23)CC1'
    #sml = 'C[NH+](C/C=C/c1ccco1)CCC(F)(F)F'  
    sml = 'c1ccccc1C'

    mols = []
    mmm = Chem.MolFromSmiles(sml)
    mols.append(mmm)

    mols.append(Chem.MolFromSmiles('P'))

    #mmm = Chem.AddHs(mmm)
    #mols.append(mmm)

    maxCuts = 2
    #maxCuts = 3

    fr = rdMMPA.FragmentMol(mmm, maxCuts = maxCuts, resultsAsMols = True, maxCutBonds = 50)
    #fr = rdMMPA.FragmentMol(mmm, pattern="[*]!@!=!#[!#1]", maxCuts=1, resultsAsMols=True, maxCutBonds=50)
    
    for f in fr:
        ff = Chem.GetMolFrags(f[1], asMols=True)
        #print(Chem.MolToSmiles(ff[0]), Chem.MolToSmiles(Chem.RemoveHs(ff[0])))
        #print(Chem.MolToSmiles(ff[1]), Chem.MolToSmiles(Chem.RemoveHs(ff[1])))
        mols.append(ff[0])
        mols.append(ff[1])

    mols.append(Chem.MolFromSmiles('P'))
    mols.append(Chem.MolFromSmiles('P'))

    #fr = rdMMPA.FragmentMol(mmm, pattern="[*]!@!=!#[!#1]", maxCuts=1, resultsAsMols=True, maxCutBonds=50)
    #fr = rdMMPA.FragmentMol(mmm, maxCuts=maxCuts, pattern='[#6+0;!$(*= , #[!#6])]! @!=!#[*]', resultsAsMols=True, maxCutBonds=50)
    #for f in fr:
    #    ff = Chem.GetMolFrags(f[1], asMols=True)
    #    #print(Chem.MolToSmiles(ff[0]), Chem.MolToSmiles(Chem.RemoveHs(ff[0])))
    #    #print(Chem.MolToSmiles(ff[1]), Chem.MolToSmiles(Chem.RemoveHs(ff[1])))
    #    mols.append(ff[0])
    #    mols.append(ff[1])


    Draw.MolsToGridImage(mols, molsPerRow = 8,).show()

    return 

if __name__ == '__main__':
    #test()

    test_sml()