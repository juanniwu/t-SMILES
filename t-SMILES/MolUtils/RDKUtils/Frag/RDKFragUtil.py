import re
import pandas as pd

import rdkit.Chem as Chem

from MolUtils.RDKUtils.Utils import RDKUtils

from Tools.StringUtils import StringUtils

import MolUtils.datamol.datamol as dm
dm.disable_rdkit_log()


from enum import Enum
class Fragment_Alg(Enum):
    Vanilla         = 0
    JTVAE           = 1          
    BRICS_Base      = 2  #only Brics
    BRICS           = 3  #Brics with other breaks, r_link, etc 
    Recap           = 4           
    MMPA            = 5            
    Scaffold        = 6        
    JTVAE_DY        = 7     
    BRICS_DY        = 8  #Brics_Dummy
    MMPA_DY         = 9  #MMPA_Dummy
    Scaffold_DY     = 10 #Scaffold_Dummy
    eMolFrag        = 11  
    RBrics_DY       = 12

class JointPiece():
    def __init__(self, atom_idx, atom_smarts, nbr_idx, nbr_atom, nbr_atom_env, bond_type, joint_idx_nbr, joint_piece) -> None:
        self.atom_idx = atom_idx
        #self.atom_symbol = atom_symbol
        self.atom_smarts = atom_smarts  

        self.nbr_idx =  nbr_idx
        self.nbr_atom = nbr_atom

        self.bond_type = bond_type
        self.nbr_atom_env = nbr_atom_env  

        self.joint_idx_nbr = joint_idx_nbr  

        self.joint_piece = joint_piece

        return       

class RDKFragUtil:
    dummy_num = 0
    dummy_char = '*'
    dummy_mol = Chem.MolFromSmiles('[*]')

    def get_atomenv_table():
        s_file = [r'D:\ProjectTF\RawData\AtomEnv\MIT_mixed\MIT_mixed_atomenv_r0.csv',
                  r'D:\ProjectTF\RawData\AtomEnv\MIT_mixed\MIT_mixed_atomenv_r1.csv',
                  r'D:\ProjectTF\RawData\AtomEnv\MIT_mixed\MIT_mixed_atomenv_r2.csv'
                 ]
        atomenv_dict = []
        for sf in s_file:
            print(sf)
            sub_ae_dict = {}

            df = pd.read_csv(sf, squeeze=True, delimiter=',',header = None, skip_blank_lines = True) 
            items = list(df.values)

            for item in items: 
                sub_ae_dict[(item[0],item[1])] = item[3]

            atomenv_dict.append(sub_ae_dict)

        return atomenv_dict


    def get_dummy_atomenv(frag_mol, dummy_donds, radius = 3):          
        dummy_atomenv = {}

        try:
            for atom in frag_mol.GetAtoms(): 
                atom_idx = None
                if  atom.HasProp("atom_idx"):
                    atom_idx = int(atom.GetProp("atom_idx"))

                if atom_idx is None:
                    continue

                atomID = atom.GetIdx()

                smarts_list = []
                for bond in dummy_donds:
                    if atom_idx in bond[0]:
                        for rds in range(0, radius ):
                            smarts = RDKitUtils.getSmarts(frag_mol, atomID, radius = rds, include_H = True)

                            smarts_list.append(smarts)   
                            
                        dummy_atomenv[atom_idx] = smarts_list
                        break 
        except Exception as e:
            print('[RDKFragUtil.get_dummy_atomenv].Exception:', e.args)

        return dummy_atomenv


    def score_atomenv(atomenv1, atomenv2, atomenv_dict, radius = 3):
        score = 0
        
        for i in range(0, radius):
            ae1 = atomenv1[i]
            ae2 = atomenv2[i]

            if ae2 is None or ae2 is None:
                s = 0.9988                
            else:
                s = atomenv_dict[i].get((ae1, ae2))
                if s is None:
                    s = atomenv_dict[i].get((ae2, ae1))
                    if s is None:
                        s = 0.9999

            score += round(s,4) * 10**(4*(radius - i))            

        return score


    def get_dummy_bond_pair(fraga, fragb):
        bond_ids = set()
        nba = None
        nbb = None
        for a in fraga.GetAtoms():
            if a.GetSymbol() == "*":
                nei_ids = set(na.GetProp('atomNote') for na in a.GetNeighbors())
                if len(nei_ids) == 1:
                    nba = list(nei_ids)[0]

        for a in fragb.GetAtoms():
            if a.GetSymbol() == "*":
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


    def GetMolFrags(mol, break_bonds, addDummies = True):

        n_cuts = len(break_bonds)
        n_atoms = mol.GetNumAtoms()

        dummyLabels = []
        for i in range(n_cuts):
            dummyLabels.append((i + 1,i + 1))

        if addDummies:
            broken = Chem.FragmentOnBonds(mol, bondIndices = break_bonds, addDummies=True, 
                                          dummyLabels = dummyLabels
                                          )
        else:
            broken = Chem.FragmentOnBonds(mol, bondIndices = break_bonds, addDummies=False)

        fragsMolAtomMapping = []
        try:
            frags_mol = Chem.GetMolFrags(broken, asMols=True, fragsMolAtomMapping = fragsMolAtomMapping)   #somtimes exception:('non-ring atom 1 marked aromatic',)         
        except Exception as e:
            print('[RDKFragUtil.Chem.GetMolFrags].exception:', e)
            raise Exception(e)

        frags_sml = [] 
        frags_smarts = []
        for fg in frags_mol:  
            try:
                Chem.Kekulize(fg)
            except Exception as e:
                print('[RDKFragUtil.Chem.Kekulize(fg)].exception:can not Kekulize mole!')
                raise Exception(e)

            sfg = Chem.MolToSmiles(fg, kekuleSmiles = True) #, kekuleSmiles = True

            frags_smarts.append(sfg)
            frags_sml.append(sfg)

        for i, sml in enumerate(frags_sml):
            new_str = StringUtils.replae_dummy_id(sml)                
            frags_sml[i] = new_str

        motifs_aidx = fragsMolAtomMapping
                        
        max_atoms = max(max(frag) for frag in fragsMolAtomMapping)
        dummy_atoms = ['*'] * (max_atoms - n_atoms + 1)

        return frags_mol, frags_smarts, frags_sml, motifs_aidx, dummy_atoms

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
    
    def cut_is_valid(frag_smls):
        # a patch to fix RDKit 
        is_valid = True
        pattern = re.compile(r"\*")

        for sml in frag_smls:
            indices_object = re.finditer(pattern = pattern, string = sml)
            indices = [index.start() for index in indices_object]

            for idx in indices:
                if idx == len(sml) -1 or sml[idx + 1] != ']':        
                    is_valid = False
                    break

            if not is_valid:
                break

        return is_valid

    def get_atomenv(sml, atomID = 0, radius = 4, label = True):
        try:
            mol = Chem.MolFromSmiles(sml)

            if mol is None:
                mol = Chem.MolFromSmarts(sml)

           
            if mol is not None:                
                smarts_list = []
                for rds in range(0, radius):  #radius shoudl be 1 based
                    smarts = None

                    if label:
                        smarts = RDKUtils.getSmarts(mol, atomID, radius = rds, include_H = True)  #if radius>0: get AE
                    else:
                        atomMap = {}
                        env = Chem.FindAtomEnvironmentOfRadiusN(mol,radius = rds, rootedAtAtom = atomID, 
                                                    )
                        amap={}
                        ae_mol= Chem.PathToSubmol(mol,env,atomMap = amap)
                        if ae_mol is not None:
                            smarts = Chem.MolToSmiles(ae_mol, isomericSmiles = False, canonical = True, kekuleSmiles = True)

                    if smarts is not None:
                        smarts_list.append(str(smarts)) 
                    else:
                        smarts_list.append(None)           
            else:
                print('[RDKAssembling.get_atomenv]: mol is None:', sml)
                for rds in range(0, radius ):               
                    smarts_list.append(None) 
        except Exception as e:
            print('[RDKAssembling.get_atomenv].Exception:',e.args)
            print('[RDKAssembling.get_atomenv].Exception:',sml)
            for rds in range(0, radius ):               
                smarts_list.append(None) 

        return smarts_list

    def mol_add_atom(base_mol, insert_pos, atom, bond_type):
        try: 
            e_mol = Chem.RWMol(base_mol)

            pos = e_mol.AddAtom(Chem.Atom(atom.GetAtomicNum()))
            e_mol.AddBond(insert_pos, pos, bond_type)
           
            Chem.SanitizeMol(e_mol)  

            e_mol.GetMol()
            e_mol = Chem.MolFromSmiles(Chem.MolToSmiles(e_mol, kekuleSmiles = False))          
        except Exception as e:
            print(e.args)
            return None

        return e_mol

    def mol_replace_atom(base_mol, idx, newAtom):
        try: 
            e_mol = Chem.RWMol(base_mol)

            pos = e_mol.ReplaceAtom(idx, newAtom) 
            
            Chem.SanitizeMol(e_mol) 

            e_mol.GetMol()
            e_mol = Chem.MolFromSmiles(Chem.MolToSmiles(e_mol))         

        except Exception as e:
            print(e.args)
            return None

        return e_mol

    def get_dummy_info(mol):
        nbr_bond_type = []
        dummy_points_base = []

        min_atom_idx = float('inf')

        try:
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 0:  
                    atom_idx = atom.GetIdx()
                    atom_smarts = atom.GetSmarts()
                    dummy_points_base.append(atom_idx)
                    if atom_idx < min_atom_idx:
                        min_atom_idx =  atom_idx 

                    bonds = atom.GetBonds()
                    for bond in bonds:
                        nbr_atom = bond.GetOtherAtom(atom)
                        nbr_idx = bond.GetOtherAtomIdx(atom.GetIdx())
                        nbr_atnum = bond.GetOtherAtom(atom).GetAtomicNum()
                        bond_type = bond.GetBondType()

                        joint_piece = RDKFragUtil.mol_add_atom(base_mol = RDKFragUtil.dummy_mol, 
                                                                insert_pos = 0, 
                                                                atom = nbr_atom, 
                                                                bond_type = bond_type,
                                                             )
                        nbr_atom_env = RDKFragUtil.get_atomenv(sml = Chem.MolToSmiles(mol),
                                                                 atomID = nbr_idx, 
                                                                 radius = 3
                                                                 )
                        if atom_idx < nbr_idx:
                            joint_idx_nbr = atom_idx
                        else:
                            joint_idx_nbr = nbr_idx

                        piece = JointPiece(atom_idx     = atom_idx, 
                                           atom_smarts  = atom_smarts,
                                           nbr_idx      = nbr_idx,
                                           nbr_atom     = nbr_atom,
                                           nbr_atom_env = nbr_atom_env,
                                           bond_type    = bond_type,
                                           joint_idx_nbr= joint_idx_nbr,
                                           joint_piece  = joint_piece,
                                           )
                        nbr_bond_type.append(piece)

        except Exception as e:
            print('[RDKFragUtil.get_dummy_info].Exception:', e.args)

        return nbr_bond_type

    def encode_dummy_ids(bfs_frags):  #give ids for dummy atoms
        #this is only a patch, which is based on bfs, so sometimes, it's reordered
        pair_pos = 1
        pos = 2

        last_dummy = None
        dummy_stack = []

        for i, frag in enumerate(bfs_frags):
            mol = Chem.MolFromSmiles(frag)
            dummy_info = RDKFragUtil.get_dummy_info(mol)

            if len(dummy_stack) > 0:
                last_dummy = dummy_stack.pop(0)

            match_idx = 0
            if last_dummy is not None:
                for j, dmy in enumerate(dummy_info):
                    if last_dummy.bond_type == dmy.bond_type:
                        match_idx = j                        
                        break

            if i == 0:
                dummy_info[0].atom_smarts = '[1*]'
                dummy_stack.append(dummy_info[0])

            frag = list(frag)
            cpy = frag.copy()

            cnt = 0
            last_pos = pair_pos
            flag = True
            for k,c in enumerate(cpy):
                if c is '*':
                    if cnt == match_idx:
                        if last_dummy is not None:
                            c = last_dummy.atom_smarts
                        else:
                            c = f'[{last_pos}*]'
                        if i > 0:
                            pair_pos +=1
                    else:
                        c = f'[{pos}*]'
                        pos += 1

                        dummy_info[cnt].atom_smarts = c
                        dummy_stack.append(dummy_info[cnt])                        

                    frag[k] = c
                    cnt +=1

            frag = ''.join(frag)
            bfs_frags[i] = frag

        return bfs_frags

    def fix_mol(smls, standardize = True):
        new_smls = None
        try:
            mol = dm.to_mol(smls)
            if mol is not None:
                mol = dm.fix_mol(mol)
                mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
                if standardize:
                    new_smls = dm.standardize_smiles(dm.to_smiles(mol)) 
                else:
                    new_smls =  Chem.MolToSmiles(mol)   
        except Exception as e:
            print('[RDKFragUtil.fix_mol].exception:', e.args)
            new_smls = Chem.MolToSmiles(Chem.MolFromSmiles(smls))

        return new_smls

    def get_longest_mols(mols):
        try:
            mols = [RDKUtils.remove_atommap_info_mol(mol) for mol in mols]

            if len(mols) == 1:
                return [Chem.MolToSmiles(mols[0])], mols

            max_atoms = 0
            cnts = []

            cands_smls = []
            cands_mols = []

            for mol in mols:
                n_atm = mol.GetNumAtoms()
                cnts.append(n_atm)
                if n_atm > max_atoms:
                    max_atoms = n_atm
                 
            for i, cnt in enumerate(cnts):                    
                if max_atoms == cnt:
                    mol = mols[i]
                    sml = Chem.MolToSmiles(mol)
                    if sml not in cands_smls:                           
                        cands_smls.append(sml)
                        cands_mols.append(mol)  

            return cands_smls, cands_mols

        except Exception as e:
            print('[RDKAssembling.get_longest_mols].exception:', e.args)
            return None, None

    def verify_candidates(candidates):
        can_smls = []
        tgt_smls = []
            
        cands = candidates        
        if len(candidates) > 0:
            for item in candidates:
                can_smls.append(item[0])    

            n_dmys = [s.count('*') for s in can_smls]
            min_dmys = min(n_dmys)

            for s in can_smls:
                if min_dmys == s.count('*'):
                    tgt_smls.append(s)

            tgt_smls = list(set(tgt_smls))
            tgt_smls.sort()

        return tgt_smls


