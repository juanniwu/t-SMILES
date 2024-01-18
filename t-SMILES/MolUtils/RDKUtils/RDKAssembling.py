import numpy as np
import random

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdmolops

from rdkit import rdBase
rdBase.DisableLog("rdApp.info")

from MolUtils.RDKUtils.Utils import RDKUtils
from MolUtils.RDKUtils.Frag.RDKFragUtil import RDKFragUtil, JointPiece
from Tools.StringUtils import StringUtils

class RDKAssembling():

    def get_atomenv_table():
        atomenv_dict = RDKFragUtil.get_atomenv_table()
        return atomenv_dict

    def canonicalize(sml, clear_stereo=False):
        mol = Chem.MolFromSmiles(sml)

        if clear_stereo:
            Chem.RemoveStereochemistry(mol)

        return Chem.MolToSmiles(mol, isomericSmiles = True)

    def mol_to_smiles(mol):
        smi = Chem.MolToSmiles(mol, isomericSmiles = True)
        return RDKAssembling.canonicalize(smi)

    def mol_from_smiles(sml):
        sml = RDKAssembling.canonicalize(sml)
        return Chem.MolFromSmiles(sml)

    def mol_to_graph_data(mol):
        A = rdmolops.GetAdjacencyMatrix(mol)
        node_features, edge_features = {}, {}

        bondidxs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]

        for idx in range(A.shape[0]):
            atomic_num = mol.GetAtomWithIdx(idx).GetAtomicNum()
            node_features[idx]["label"] = int(atomic_num)

        for b1, b2 in bondidxs:
            btype = mol.GetBondBetweenAtoms(b1, b2).GetBondTypeAsDouble()
            edge_features[(b1, b2)]["label"] = int(btype)

        return A, node_features, edge_features
   
    def strip_dummy_atoms(mol):
        hydrogen_mol = RDKAssembling.mol_from_smiles('[H]')

        mols = Chem.ReplaceSubstructs(mol, RDKFragUtil.dummy_mol, hydrogen_mol, replaceAll=True)
        mol = Chem.RemoveHs(mols[0])
        return mol

    def has_dummy_atom(mol):
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                return True
        return False


    def count_dummies(mol):
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                count += 1
        return count

    def assembling(base_mol, neighbor):     
        print(Chem.MolToSmiles(neighbor))
        print(Chem.MolToSmiles(base_mol))

        mol_list = []
        new_mol = None
        n_base = len(base_mol.GetAtoms())
        itry = 0

        try:
            while itry < n_base:
                atom_index = [x for x in range(len(base_mol.GetAtoms()))]
                idx_base = random.sample(atom_index, 2)
                posi1 = idx_base[0]
                posi2 = idx_base[1]

                for i in range(len(neighbor.GetAtoms())):
                   new_mol = RDKAssembling.assemb_fragment(base_mol, posi1, neighbor, i)
                   if new_mol is not None:
                       mol_list.append(new_mol)
                       print(Chem.MolToSmiles(new_mol))
                       itry +=1
                       break
                itry += 1

        except Exception as e:
            print(e.args)

        return mol_list

    def assemb_fragment(base_mol, idx_base, neighbor, idx_nb):
        try:
            e_mol = Chem.RWMol(base_mol)
            n_atoms = len(e_mol.GetAtoms())

            for a in neighbor.GetAtoms():
                e_mol.AddAtom(Chem.Atom(a.GetAtomicNum()))

            bonds = list(zip(*np.where(Chem.GetAdjacencyMatrix(neighbor))))
            for i, j in bonds:
                i = int(i)
                j = int(j)

                if i<j:
                    bond = neighbor.GetBondBetweenAtoms(i,j)
                    e_mol.AddBond(i + n_atoms, j + n_atoms, bond.GetBondType())

            e_mol.AddBond(idx_base, n_atoms + idx_nb, Chem.BondType.SINGLE) #???
                 
            Chem.SanitizeMol(e_mol)    

            Chem.Kekulize(e_mol)  

        except Exception as e:
            print(e.args)
            return None

        return e_mol

    def find_c_pos(mol):
        pos = 0
        cpos = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum()  == 6 and atom.GetTotalNumHs() > 0:
                cpos.append(atom.GetIdx())
        if len(cpos) > 0:
            pos = random.choice(cpos)

        return pos

    def mol_add_dummy(base_mol, insert_pos = 0, show = False):
        bond_type = Chem.rdchem.BondType.SINGLE 
        try: 
            e_mol = Chem.RWMol(base_mol)

            pos = e_mol.AddAtom(Chem.Atom(0)) 
            e_mol.AddBond(insert_pos, pos, bond_type)
             
            Chem.SanitizeMol(e_mol)  

            e_mol.GetMol()
            e_mol = Chem.MolFromSmiles(Chem.MolToSmiles(e_mol, kekuleSmiles = False))        
            

        except Exception as e:
            print(e.args)
            return None

        return e_mol

    def correct_Kekulize(mol):  #
        try:
            Chem.Kekulize(mol)
        except Exception as e:
            sml = Chem.MolToSmiles(mol, kekuleSmiles = False)
            new_sm = sf.decoder(sf.encoder(sml))
            new_mol = Chem.MolFromSmiles(new_sm)
            mol = new_mol

        return 

    def correct_ring(sml):
        snew_sm = sf.decoder(sf.encoder(sml))
        return snew_sm


    def remove_dummy_ids(smarts):
        smarts = StringUtils.replae_dummy_id(smarts)
        return smarts
    

    def get_matched_points(base_dummys, nbr_dummys, 
                           match_alg = 'match_dummy_idx',  #match with atom ids
                           #match_alg = 'match_atomenv',   #math with atom env
                           #match_alg = 'match_all',       #without dummy id, match all possible dummy atoms
                           dif_mol = True,
                           ):
        match_idx = []
        score = 0        

        try:
            if len(nbr_dummys) == 0: 
                for value in base_dummys:
                    match_idx.append((value, None))
            elif match_alg == 'match_atomenv':
                for value in base_dummys:
                    bnd_type_base   = value.bond_type
                    smarts_base     = value.atom_smarts
                    pos             = smarts_base.find('*')
                    smarts_base     = smarts_base[0:pos]
                    atomenv_base    = value.nbr_atom_env

                    for value_nb in nbr_dummys:
                        bnd_type_nb = value_nb.bond_type
                        smarts_nb   = value_nb.atom_smarts
                        pos         = smarts_nb.find('*')
                        smarts_nb   = smarts_nb[0:pos]
                        atomenv_nbr = value_nb.nbr_atom_env

                        if bnd_type_base == bnd_type_nb:
                            score = RDKFragUtil.score_atomenv(atomenv_base, atomenv_nbr, RDKAssembling.get_atomenv_table(), radius = 3)
                            match_idx.append((value, value_nb, score))
            elif match_alg == 'match_dummy_idx': 
                for value in base_dummys:
                    bnd_type_base   = value.bond_type
                    smarts_base     = value.atom_smarts
                    pos = smarts_base.find('*')
                    smarts_base = smarts_base[0:pos]

                    for value_nb in nbr_dummys:
                        bnd_type_nb = value_nb.bond_type
                        smarts_nb   = value_nb.atom_smarts
                        pos = smarts_nb.find('*')
                        smarts_nb = smarts_nb[0:pos]

                        if bnd_type_base == bnd_type_nb and smarts_base == smarts_nb:
                            if dif_mol:                            
                                match_idx.append((value, value_nb, score))                                  
                            elif value.atom_idx != value_nb.atom_idx:
                                match_idx.append((value, value_nb, score))
                            break
            else: 
                for value in base_dummys:
                    bnd_type_base = value.bond_type

                    for value_nb in nbr_dummys:
                        bnd_type_nb = value_nb.bond_type

                        if bnd_type_base == bnd_type_nb:
                            match_idx.append((value, value_nb, 0)) 

                if len(match_idx) == 0:  
                    for value in base_dummys:
                        for value_nb in nbr_dummys:
                                match_idx.append((value, value_nb, score))

        except Exception as e:
            print('[RDKAssembling.get_matched_points].Exception:', e.args)

        return match_idx

    def join_singel_frag(mol_base, base_info:JointPiece, nb_info:JointPiece):
        n_atoms_base = mol_base.GetNumAtoms()
        final_mol = mol_base
        try:            
            show = True
            #show = False

            if show:
                RDKUtils.add_atom_index(mol_base)
                mols = [mol_base]
                Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(500, 500),  
                                     legends=[Chem.MolToSmiles(m, kekuleSmiles = False) for m in mols]).show()

            ed_merged_mol = Chem.EditableMol(mol_base)

            pt_base = base_info.nbr_idx
            pt_nb   = nb_info.nbr_idx

            ed_merged_mol.RemoveAtom(nb_info.atom_idx)

            ed_merged_mol.AddBond(pt_base, pt_nb, order = base_info.bond_type)

            ed_merged_mol.RemoveAtom(base_info.atom_idx)
            if show:
                final_mol = ed_merged_mol.GetMol()            
                Draw.MolsToGridImage([final_mol], molsPerRow=3, subImgSize=(500, 500), 
                                     legends=[Chem.MolToSmiles(m, kekuleSmiles = False) for m in [final_mol]]).show()


            final_mol = ed_merged_mol.GetMol()

            Chem.SanitizeMol(final_mol)   

            final_mol = Chem.MolFromSmiles(Chem.MolToSmiles(final_mol))            
       
            if show:
                mols.append(final_mol)
                Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(500, 500)).show()

        except Exception as e:
            print('[RDKAssembling.join_singel_frag].exception:', e.args)
            print('[RDKAssembling.join_singel_frag].exception:', Chem.MolToSmiles(mol_base))
            msg = e.args
            
        return final_mol

    def join_substructs(mol_base, neighbour, base_info:JointPiece, nb_info:JointPiece):
        n_atoms_base = mol_base.GetNumAtoms()
        n_atoms_nb = neighbour.GetNumAtoms()

        final_mol = mol_base

        try:        
            merged_mol = Chem.CombineMols(mol_base, neighbour)

            n_atoms_mg = merged_mol.GetNumAtoms()   

            show = False

            if show:
                RDKUtils.add_atom_index(mol_base)
                RDKUtils.add_atom_index(neighbour)
                RDKUtils.add_atom_index(merged_mol)
                mols = [mol_base, neighbour,  merged_mol]
                Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(500, 500),  
                                     legends=[Chem.MolToSmiles(m, kekuleSmiles = False) for m in mols]).show()


            ed_merged_mol= Chem.EditableMol(merged_mol)

            pt_base = base_info.nbr_idx
            pt_nb = nb_info.nbr_idx + n_atoms_base 

            ed_merged_mol.AddBond(pt_base, pt_nb, order = base_info.bond_type)

            ed_merged_mol.RemoveAtom(base_info.atom_idx)
         

            ed_merged_mol.RemoveAtom(nb_info.atom_idx + n_atoms_base - 1)

            final_mol = ed_merged_mol.GetMol()

            Chem.SanitizeMol(final_mol)   

            final_mol = Chem.MolFromSmiles(Chem.MolToSmiles(final_mol))            

            if show:
                mols.append(final_mol)
                Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(500, 500)).show()
        
        except Exception as e:
            print('[RDKAssembling.join_substructs].exception:', e.args)
            print('[RDKAssembling.join_substructs].exception:', Chem.MolToSmiles(mol_base))
            print('[RDKAssembling.join_substructs].exception:', Chem.MolToSmiles(neighbour))
            msg = e.args
            
        return final_mol
    
    def rdkit_ReplaceSubstructs(base_mol, neighbour, base_info:JointPiece, nb_info:JointPiece):
        try:
            pt = nb_info.atom_idx
            ed = Chem.EditableMol(neighbour) # 
            ed.RemoveAtom(pt)
            new_nbr = ed.GetMol()              
        
            query = RDKFragUtil.dummy_mol
            joint_pt_nb = nb_info.joint_idx_nbr         
              
            show = False

            if show:
                RDKUtils.add_atom_index(base_mol)
                RDKUtils.add_atom_index(neighbour)
                RDKUtils.add_atom_index(new_nbr)

                mols = [base_mol, neighbour, new_nbr]
                Draw.MolsToGridImage(mols,subImgSize=(500, 500), legends=[Chem.MolToSmiles(m) for m in mols]).show()

            try:
                combined = Chem.ReplaceSubstructs(
                            mol           = base_mol ,   
                            query         = query, 
                            replacement   = new_nbr,                  
                            replaceAll    = False,                   
                            replacementConnectionPoint = joint_pt_nb, 
                            useChirality  = False                      
                            )      

            except Exception as e:
                print('[RDKAssembling.rdkit_ReplaceSubstructs.[Chem.ReplaceSubstructs]].exception:', e.args)


            if show:
                mols.extend(combined)
                Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(500, 500), legends=[Chem.MolToSmiles(m) for m in mols]).show()
  
            cands = []
            joined_smls = []
            if combined is not None:
                for join in combined:
                    atom_num = join.GetAtomWithIdx(base_info.atom_idx).GetAtomicNum()

                    if RDKFragUtil.dummy_num == atom_num:
                        continue

                    Chem.SanitizeMol(join)
                    joined_smls.append(Chem.MolToSmiles(join)) 

            joined_smls = list(set(joined_smls))
            joined_smls.sort()


            for sml in joined_smls:
                joined_mol = Chem.MolFromSmiles(sml)
                if joined_mol is not None:
                    cands.append(joined_mol)

        except Exception as e:
            print('[RDKAssembling.rdkit_ReplaceSubstructs].Exception:', e.args)
            print('[RDKAssembling.rdkit_ReplaceSubstructs].base_mol:', Chem.MolToSmiles(base_mol))
            print('[RDKAssembling.rdkit_ReplaceSubstructs].neighbour:', Chem.MolToSmiles(neighbour))
          
        if show:
            mols = [base_mol, neighbour]
            mols.extend(cands)
            Draw.MolsToGridImage(mols, subImgSize=(500, 500), legends=[Chem.MolToSmiles(m) for m in mols]).show()

        return cands

    #-----------------------------------------------
    def check_dummy_atom(sml_base, sml_nbr):
        find1 = StringUtils.find_dummy_fmt(sml_base, '[c*]')
        find2 = StringUtils.find_dummy_fmt(sml_base, '[*]')
        find3 = StringUtils.find_dummy_fmt(sml_base, '*')

        f_and = find1 and find2 and find3
        f_or = find1 or find2 or find3

        f12 = find1 and find2
        f13 = find1 and find3
        f23 = find2 and find3

        if not f_or:
            no_dummy = True
        else:
            no_dummy = False

        if f_or and f_and: 
            fixed = True        
        elif not f_or:
            fixed = False
        elif not f12 and not f13 and not f23:
            fixed = False
        else:
            fixed = True

        return fixed, no_dummy
    #--------------------------------------------------------------------------------

    def combine_frages_info(base_mol, neighbour, match_joints, replace_alg):
        cands_mols = []
        cands_smls = []
        scores = []

        for match_id in match_joints:
            base_info = match_id[0]
            nb_info = match_id[1]

            query = base_info.joint_piece  

            max_atoms = 0
            try:
                has_sub = base_mol.HasSubstructMatch(query)
                if has_sub:
                    if replace_alg =='join_sub' :
                        join_mol = RDKAssembling.join_substructs(base_mol,
                                                                    neighbour,
                                                                    base_info,
                                                                    nb_info, 
                                                                    )

                        join_mol = RDKUtils.remove_atommap_info_mol(join_mol)
                        joined_sml = Chem.MolToSmiles(join_mol)

                        if joined_sml not in cands_smls:  
                            cands_smls.append(joined_sml)

                            joined_mol = Chem.MolFromSmiles(joined_sml)
                            if joined_mol is not None:
                                cands_mols.append(joined_mol)    
                                scores.append(match_id[2])                     
                    else:
                        joined_mols = RDKAssembling.rdkit_ReplaceSubstructs(base_mol, neighbour, base_info, nb_info)  
                        smls, cands = RDKFragUtil.get_longest_mols(joined_mols)
                        for i, sml in enumerate(smls):
                            if sml not in cands_smls:
                                cands_smls.append(sml)
                                cands_mols.append(cands[i]) 
                                scores.append(0)
                        
            except Exception as e:
                print('[RDKAssembling.join_substructs-rdkit_ReplaceSubstructs].exception:', e.args)
                continue                

        return cands_smls, cands_mols, scores


    def assemb_mols_single(base_mol, match_alg = 'match_dummy_idx'):
        sml_base = Chem.MolToSmiles(base_mol)

        show = False
        #show = True

        if show:
            RDKUtils.add_atom_index(base_mol)
            mols = [base_mol]
            Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(500, 500),  legends=[Chem.MolToSmiles(m) for m in mols]).show()


        if match_alg != 'match_dummy_idx':
            return [base_mol], [sml_base], [0.]

        sgl_dummys = RDKFragUtil.get_dummy_info(base_mol)
        match_joints = RDKAssembling.get_matched_points(sgl_dummys, sgl_dummys, match_alg, dif_mol = False) 

        if len(match_joints) != 1:
            return [base_mol], [sml_base], [0.]
           
        cands_mols = []
        cands_smls = []
        scores = []

        try: 
            match_id = match_joints[0]
            snd_atom_smarts = match_id[0].atom_smarts  #'[4*:7]'

            snd_dummy = []
            for dm in sgl_dummys:
                pos = dm.atom_smarts.find('*')
                if pos > 0:
                    dyid = dm.atom_smarts[:pos]
                    if snd_atom_smarts.startswith(dyid): #   '[4*]'
                        snd_dummy.append(dm)

            if len(snd_dummy) == 2:
                base_info = snd_dummy[0]
                nb_info = snd_dummy[1]

                join_mol = RDKAssembling.join_singel_frag(base_mol,
                                                          base_info,
                                                          nb_info, 
                                                          )
                if join_mol is not None:
                    join_mol = RDKUtils.remove_atommap_info_mol(join_mol)
                    joined_sml = Chem.MolToSmiles(join_mol)

                    cands_mols = [join_mol]
                    cands_smls = [joined_sml]
                    scores = [match_id[2]]    
        except Exception as e:
            print('[RDKAssembling.assemb_mols_single].exception:', e.args)  
            print(sml_base)
            
        return  cands_mols, cands_smls, scores 


    def assemb_mols_dummy(base_mol, neighbour, 
                          correct_dummy = True, 
                         
                          #match_alg = 'match_dummy_idx',  #match with atom ids, only match one point
                          #match_alg = 'match_atomenv',   #math with atom env
                          match_alg = 'match_all',       #without dummy id, match all possible dummy atoms
                          
                          #replace_alg = 'join_sub',
                          replace_alg = 'rdkit_replace', #more aandidates with Chirality

                          n_candidates = -1,
                          ):        
        sml_base = Chem.MolToSmiles(base_mol)
        sml_nbr = Chem.MolToSmiles(neighbour)

        show = False
        #show = True

        if show:
            print('[assemb_mols_dummy]sml_base =', sml_base)
            print('[assemb_mols_dummy]sml_neib =',sml_nbr)

            RDKUtils.add_atom_index(base_mol)
            RDKUtils.add_atom_index(neighbour)

            mols = [base_mol, neighbour]
            Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(500, 500),  legends=[Chem.MolToSmiles(m) for m in mols]).show()
            
        if match_alg == 'match_dummy_idx' and not RDKFragUtil.cut_is_valid([sml_base, sml_nbr]):
            print(f'[match_dummy_idx] ids format dismatch, use match_all instead!')
            sml_base = StringUtils.replae_dummy_id(sml_base)
            sml_nbr = StringUtils.replae_dummy_id(sml_nbr)
            match_alg = 'match_all'


        n_dmy_base = sml_base.count('*')
        n_dmy_nbr = sml_nbr.count('*')

        if n_dmy_base == 0 :
            if correct_dummy:
                insert_pos =  RDKAssembling.find_c_pos(base_mol)
                base_mol = RDKAssembling.mol_add_dummy(base_mol, insert_pos)
            else:
                return [base_mol], [0]    

        if n_dmy_nbr == 0:
            insert_pos =  RDKAssembling.find_c_pos(neighbour)
            neighbour = RDKAssembling.mol_add_dummy(neighbour, insert_pos)

        if n_dmy_nbr < n_dmy_base: 
            temp = base_mol
            base_mol = neighbour
            neighbour = temp        
           
        if base_mol is not None and neighbour is None:
            return [base_mol], [0] 
        if base_mol is None and neighbour is None:
            return [Chem.MolFromSmiles('CC')], [0] 
        if base_mol is None and neighbour is not None:
            return [neighbour], [0] 

        if show:
            mols = [base_mol, neighbour]
            Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(500, 500),  legends=[Chem.MolToSmiles(m) for m in mols]).show()

        try:
            base_dummys = RDKFragUtil.get_dummy_info(base_mol)
            nbr_dummys = RDKFragUtil.get_dummy_info(neighbour)

            match_idxs = RDKAssembling.get_matched_points(base_dummys, nbr_dummys, match_alg, dif_mol = True) 

            if(len(match_idxs) == 0):
                print(f'[match_idxs].length is zero, use match_all instead')

                match_alg = 'match_all',
                match_idxs = RDKAssembling.get_matched_points(base_dummys, nbr_dummys, match_alg, dif_mol = True) 
                print(f'[match_idxs].length:', len(match_idxs))

            if n_candidates > 0 and n_candidates < len(match_idxs):
                ids = list(range(len(match_idxs)))
                random.shuffle(ids)
                ids = ids[0:n_candidates]
                match_joints = [match_idxs[i] for i in ids]
            else:
                match_joints = match_idxs


            cands_mols = []
            cands_smls = []
            scores = []

            if match_alg == 'match_dummy_idx' and len(match_joints) == 2:
                match_joints_tmp = [match_joints[0]]
                cands_smls, cands_mols, scores = RDKAssembling.combine_frages_info(base_mol, neighbour, match_joints_tmp, replace_alg)

                if len(cands_mols) > 0:
                    cands_mols_2nd, cands_sml_2nd, scores_2nd = RDKAssembling.assemb_mols_single(cands_mols[0], match_alg = 'match_dummy_idx')
                    if cands_mols_2nd is not None and len(cands_mols_2nd) > 0:
                        joined_mol = cands_mols_2nd[0]
                        joined_sml = cands_sml_2nd[0]
                        score      = scores_2nd[0]   
                        
                        cands_mols = [joined_mol]
                        cands_smls = [joined_sml]
                        scores = [score]    
            else:
                cands_smls, cands_mols, scores = RDKAssembling.combine_frages_info(base_mol, neighbour, match_joints, replace_alg)

        except Exception as e:
            print('[RDKAssembling.assemb_mols_dummy].exception:', e.args)

        if show:   
            print('[RDKAssembling.assemb_mols_dummy]: len of cands:', len(cands_mols))
            print(cands_smls)

            mols = [base_mol, neighbour]
            mols.extend(cands_mols)
            Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(500, 500),  legends=[Chem.MolToSmiles(m) for m in mols]).show()

        return cands_mols, scores  #


    
def test_assemb_mols_dummy():   

    #sml_base = '*N*'
    #sml_neib = '*C1(*)CCCCC1'

    #sml_base = '[2*]S(=O)(=O)CCCN1CCCCC1'
    #sml_neib = '[2*]N1CCN(S(=O)(=O)c2ccc(-c3ccc(F)cc3)s2)C([4*])C1'

    sml_base = '[3*]S(=O)(=O)CCCN1CCCCC1'
    sml_neib = '[2*]N1CCN(S(=O)(=O)c2ccc(-c3ccc(F)cc3)s2)C([4*])C1'

    #sml_base = '[1*]C(=NS([3*])(=O)=O)N[4*]'
    #sml_neib = '[2*]c1ccc([3*])c([4*])c1'

    base = Chem.MolFromSmiles(sml_base)
    neib = Chem.MolFromSmiles(sml_neib)

    if base is None or neib is None:
        print('Mol is None')

    s = Chem.MolToSmiles(base)
    s = Chem.MolToSmiles(neib)

    joined, scores = RDKAssembling.assemb_mols_dummy(base, neib,
                                                    match_alg = 'match_dummy_idx',  #match with atom ids, only match one point
                                                    #match_alg = 'match_atomenv',   #math with atom env
                                                    #match_alg = 'match_all',       #without dummy id, match all possible dummy atoms
                          
                                                    replace_alg = 'join_sub',
                                                    #replace_alg = 'rdkit_replace', #more aandidates with Chirality

                                                    n_candidates = 1,
                                                    )
    for i, join_mol in enumerate(joined):
        joined_sml =  Chem.MolToSmiles(join_mol)
        fix_smls = RDKFragUtil.fix_mol(joined_sml)
        if fix_smls is not None:
            joined[i] = Chem.MolFromSmiles(fix_smls)

    #print(sml_base)
    #print(sml_neib)
    print([Chem.MolToSmiles(join) for join in joined])
    #print(Chem.MolToSmiles(joined))


    return

if __name__ == '__main__':
    n_samples = 1
    for i in range(n_samples):
        test_assemb_mols_dummy()
