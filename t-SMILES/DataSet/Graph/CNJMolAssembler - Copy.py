import numpy as np
import pandas as pd
from tqdm import tqdm

import random
import copy

import networkx as nx

import rdkit
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers

from DataSet.STDTokens import CTokens, STDTokens_Frag_File
from DataSet.Graph.CNJTMol import CNJTMolTreeNode, CNJTMolTree
from DataSet.Graph.CNJMolUtil import CNJMolUtil
from DataSet.JTNN.MolTree import MolTreeNode, MolTreeUtils
from DataSet.JTNN.ChemUtils import ChemUtils
from MolUtils.RDKUtils.Frag.RDKFragUtil import RDKFragUtil,CODE_Alg
from MolUtils.RDKUtils.RDKAssembling import RDKAssembling
from MolUtils.RDKUtils.Utils import RDKUtils


def CNJASM_HParam():
    config = {
        #alg               = 'CALG_TSSA', #JTVAE: without dummy atom
        #alg               = 'CALG_TSDY', #with dummy atom, without dummy id
        #alg               = 'CALG_TSID', #with dummy atom, with dummy id
        #alg               = 'CALG_TSIS', #with dummy atom, with dummy id, based on amt, bfs
        #alg               = 'BRICS', #this should not be used now as linking to parts may wrong
        #alg               = 'MM',    #candidates = node.smiles

        #assemble_alg     = 'JTVAE_decode',   #JTVAE_decode: ChemUtils.decode_moltree
        #assemble_alg      = 'leaf_first',     #leaf_first: build a candidadate tree
        #assemble_alg     = 'group_first',     #group_first: randomly selece one candidate as target  
                                                                
        #always use candidate_tree to get candidate
        #candidate_mode    = 'random_one',    #random_one: randomly selece one candidate as target 
        #candidate_mode    = 'candidate_tree', #candidate_tree: build a candidadate tree
                                                      
        #final_select_alg  = 'random',      #random: randomly select on candidate as final output
        #final_select_alg  = 'plogp',       #plogp: maximum penalized_logp
        #final_select_alg  = 'candidates',  #return all from candidate_tree                                                              
        #final_select_alg  = 'goal_valsartan_smarts',       #return a from GoalDirectedBenchmark which is defined in from External.Guacamol.guacamol.standard_benchmarks                                                              

        #
        #match_alg = 'match_dummy_idx',  #match with atom ids, only match one point
        #match_alg = 'match_atomenv',   #math with atom env
        #match_alg = 'match_all',       #without dummy id, match all possible dummy atoms
                          
        #replace_alg = 'join_sub',
        #replace_alg = 'rdkit_replace', #more aandidates with Chirality

        #------------
        'alg'               : 'CALG_TSDY',  #hparam['alg'] == 'CALG_TSDY' or hparam['alg'] == 'CALG_TSID' or hparam['alg'] == 'CALG_TSSA' or hparam['alg'] == 'CALG_TSIS':
        'assemble_alg'      : 'leaf_first',
        'candidate_mode'    : 'candidate_tree',
        'final_select_alg'  : 'random',
        
        'match_alg'         : 'match_all',
        'replace_alg'       : 'rdkit_replace',

        #---------------------------------------------------
        #for none_dummy

        #'alg                : 'JTVAE',
        #'assemble_alg'      : 'JTVAE_decode',
        #'candidate_mode'    : 'candidate_tree',
        #'final_select_alg'  : 'random',

        #-------------------------------------------------
        #'cand_alg'          : 'atomenv',
        'cand_alg'          : 'random',

        'n_candidates'      : 25,  # candidates gets score than 99%
        #'n_candidates'      : 1,  # 

        'use_stereo'        : False,
        'pMean'             : None,

    }
    return config


class CNJMolAssembler:
    def get_hparam(asm_alg = 'CALG_TSDY'):
        #alg='No_Dummy'     'CALG_TSSA'
        #alg='Dummy',       'CALG_TSDY',
        #alg='Dummy_AEID',  'CALG_TSID',
        #alg='Dummy_AEID'   'CALG_TSIS'  #new added

        hparam = CNJASM_HParam()
        if asm_alg == 'CALG_TSDY': #'Dummy'
            hparam['alg']               = 'CALG_TSDY'
            hparam['assemble_alg']      = 'leaf_first'
            hparam['candidate_mode']    = 'candidate_tree'
            hparam['n_candidates']      = 3

            hparam['final_select_alg']  = 'random'
            #hparam['final_select_alg'] = 'goal_sitagliptin_replacement'    #----------------------16.smpo
        
            hparam['match_alg']         = 'match_all'
            hparam['replace_alg']       = 'join_sub'  
            #hparam['replace_alg']       = 'rdkit_replace'  

            hparam['cand_alg']          = 'random'
        elif asm_alg == 'CALG_TSID':  #'Dummy_AEID'
            hparam['alg']               = 'CALG_TSID'
            hparam['assemble_alg']      = 'leaf_first'
            hparam['candidate_mode']    = 'candidate_tree'
            hparam['n_candidates']      = 3
            hparam['final_select_alg']  = 'random'
            #hparam['final_select_alg']  = 'goal_sitagliptin_replacement'   #----------------------16.smpo
        
            hparam['match_alg']         = 'match_dummy_idx'
            hparam['replace_alg']       = 'join_sub'

            #hparam['cand_alg']          = 'atomenv'
            hparam['cand_alg']          = 'random'
        elif asm_alg == 'CALG_TSIS': #'Dummy_AEID'
            hparam['alg']               = asm_alg
            hparam['assemble_alg']      = 'leaf_first'
            hparam['candidate_mode']    = 'candidate_tree'
            hparam['final_select_alg']  = 'random'
            hparam['n_candidates']      = 1
        
            hparam['match_alg']         = 'match_dummy_idx'
            hparam['replace_alg']       = 'join_sub'

            #hparam['cand_alg']          = 'atomenv'
            hparam['cand_alg']          = 'random'
        else: #No_Dummy, CALG_TSSA
            hparam['alg']               = 'CALD_TSSA'
            hparam['assemble_alg']      = 'JTVAE_decode'            
            hparam['candidate_mode']    = 'random_one'            

            #hparam['n_candidates']      = 0  #default for normal rebuild
            hparam['n_candidates']      = 5   #get more candidates for Goal-directed Reconstruction 
          
           
            hparam['final_select_alg']  = 'random'        
            #hparam['final_select_alg']  = 'goal_sitagliptin_replacement'    #----------------------16.smpo

        return hparam

    def enum_assemble(node, neighbors, prev_nodes=[], prev_amap=[], hparam = CNJASM_HParam()): 
        candidates = []
        candidates_sml = []
        try:
            if hparam['alg'] ==  'CALD_TSSA':
                candidates = ChemUtils.enum_assemble(node, neighbors, prev_nodes, prev_amap)
            elif hparam['alg'] == 'Brics':
                frags = [node.smiles]
                frags_mol = [Chem.MolFromSmiles(node.smiles)]
                for nb in neighbors:
                    frags.append(nb.smiles)
                    frags_mol.append(Chem.MolFromSmiles(nb.smiles))
                    
                new_mol = RDKBrics.build_mol(frags)
                new_mol, _ = RDKAssembling.reconstruct_brics(frags_mol)
                if new_mol is not None:
                    smiles = Chem.MolToSmiles(new_mol)
                    amap = [] #like [(2, 0, 0)]
                    candidates.append((smiles, new_mol, amap))
            elif hparam['alg'] == 'CALG_TSDY' or hparam['alg'] == 'CALG_TSID' or hparam['alg'] == 'CALG_TSIS':
                frags = [node.smiles]
                base_mol = Chem.MolFromSmiles(node.smiles)

                for nb in neighbors:
                    nb_cands = []
                    if len(nb.candidates)==0:
                        nb_cands = [nb.smiles]
                    else:
                        for cd in nb.candidates:
                            nb_cands.append(cd[0])

                    nb_cands = set(nb_cands)
                    for ncd in nb_cands:
                        nb_mol = Chem.MolFromSmiles(ncd)

                        new_mol_cands, scores  = RDKAssembling.assemb_mols_dummy(base_mol, nb_mol,
                                                                                 match_alg   = hparam['match_alg'],      
                                                                                 replace_alg = hparam['replace_alg'],
                                                                                 n_candidates = -1 #hparam['n_candidates']
                                                                                 )                    
                        for i, new_mol in enumerate(new_mol_cands):
                            smiles = Chem.MolToSmiles(new_mol)
                            if smiles not in candidates_sml:
                                candidates_sml.append(smiles)
                                amap = [] #like [(2, 0, 0)]
                                candidates.append((smiles, new_mol, amap, scores[i]))
                            #break
            elif hparam['alg'] =='MM':
                candidates = [(node.smiles,Chem.MolToSmiles(node.smiles))]
            else:
                raise ValueError('[enum_assemble] raise expection:', hparam['alg'])

        except Exception as e:
            print('[CNJMolAssembler.enum_assemble].Exception: ',e.args)
            print('[CNJMolAssembler.enum_assemble].Exception.node', node.smiles)
            print('[CNJMolAssembler.enum_assemble].Exception.neighbors', neighbors[0].smiles)
            candidates = node.candidates
         
        #print('[out]:[CNJMolAssembler.enum_assemble:]', candidates)

        return candidates

    def Convert_amt_JTMoltree(vocab,
                             ctoken,
                             g_amt,  #nx Graph
                             bfs_node_map,
                             show = True,
                             ):
        moltree = CNJTMolTree(smiles = '', jtvoc = vocab, ctoken = ctoken)
         
        n_nodes = len(g_amt.nodes)
        moltree.nodes = []

        for idx in g_amt.nodes:
            node = g_amt.nodes[idx]
            smile = node['smile']

            if CNJMolUtil.is_dummy(smile):
                continue
            else:
                jtnode = MolTreeNode(smile)                 
                jtnode.idx = idx

                jtnode.wid = ctoken.STDTokens.vocab.get_index(smile)  
                moltree.nodes.append(jtnode)
                g_amt.nodes[idx]['jtnode'] = jtnode

        for idx in g_amt.nodes:
             node = g_amt.nodes[idx]
             for n in g_amt.neighbors(idx):
                if g_amt.nodes[n]['jtnode'] not in node['jtnode'].neighbors:
                    node['jtnode'].neighbors.append(g_amt.nodes[n]['jtnode'])
                if node['jtnode'] not in g_amt.nodes[n]['jtnode'].neighbors:
                    g_amt.nodes[n]['jtnode'].neighbors.append(node['jtnode'])

        for i, node in enumerate(moltree.nodes):
            node.idx = i
            node.nid = node.idx + 1
            node.is_leaf = (len(node.neighbors) == 1)
            node.candidates = []

        dummy_nodes = []
        for node in moltree.nodes: 
            if CNJMolUtil.is_dummy(node.smiles):
                node.neighbors[0].neighbors.remove(node)
                dummy_nodes.append(node)
                continue

            if node.mol is None:
                node.smiles = CNJMolUtil.valid_smiles(node.smiles, ctoken = ctoken)
                node.mol = Chem.MolFromSmiles(node.smiles)
            
            node.is_leaf = (len(node.neighbors) == 1)

        moltree.graph_nx = g_amt
        root = moltree.nodes[0]
        all_nodes = moltree.nodes

        if show:          
            GTools.show_network_g_cnjtmol(g_amt)


        return root, all_nodes, g_amt, moltree


    def Convert_to_JTMoltree(vocab,
                             ctoken,
                             bfs_binary_tree:CNJTMolTreeNode,
                             bfs_node_map,
                             show = True,
                             ):
        g = nx.Graph()
        moltree = CNJTMolTree(smiles = '', jtvoc = vocab, ctoken = ctoken)

        stack, trace = [], []
        stop = False

        bfs_ex_nodeid, bfs_ex_vocids, bfs_ex_smiles, new_vocs, bfs_ex_smarts = CNJTMolTree.get_bfs_ex(ctoken, bfs_binary_tree, extra_dummy= False)  

        bfsnode = bfs_binary_tree
        vocid = bfsnode.data['idx_voc']
        smile = bfsnode.data['smile']
        smarts = bfsnode.data['smarts']
        wid = ctoken.STDTokens.vocab.get_index(smile)  

        root_wid = wid
        root = MolTreeNode(smiles = smile, clique=[], smarts = smarts)
        root.wid = root_wid
        root.idx = bfsnode.idx
        stack.append((root, ctoken.STDTokens.vocab.get_slots(root.wid)))       

        bfs_queue = []
        bfs_queue.append(bfsnode)

        jtvae_queue = []
        jtvae_queue.append((root, MolTreeUtils.get_slots(root.smiles)))

        all_nodes = [root]
        parent_map = {}
        id_map = {}
        id_map[root.idx] = (root, MolTreeUtils.get_slots(root.smiles))  

        g.add_node(root.idx, 
                   nid = root.idx,  
                   smile = root.smiles,
                   jtnode = root,
                   smarts = root.smarts
                   )
        try:
            while len(bfs_queue) > 0 and not stop:
                bds_node = bfs_queue.pop(0)

                node_x, fa_slot = id_map[bds_node.idx]

                node_left = bds_node.left
                node_right = bds_node.right

                if node_left is not None and not CNJMolUtil.is_dummy(node_left.data['smile']):
                    idx   = node_left.idx
                    vocid = node_left.data['idx_voc']
                    smile = node_left.data['smile']
                    smarts = node_left.data['smarts']
                    wid = ctoken.STDTokens.vocab.get_index(smile)  

                    slots = MolTreeUtils.get_slots(smile)
                    fa_slot = MolTreeUtils.get_slots(node_x.smiles)
             
                    node_y = MolTreeNode(smile)

                    next_wid = wid
                    next_slots = slots

                    node_y.wid = next_wid
                    node_y.idx = idx
                    node_y.neighbors.append(node_x)
                    node_x.neighbors.append(node_y)

                    all_nodes.append(node_y)

                    stack.append((node_y, next_slots))

                    parent_map[node_y] = node_x
                    id_map[idx] = (node_y, next_slots)
                
                    g.add_edge(node_x.idx, node_y.idx)
                    g.nodes[idx]['nid'] = idx
                    g.nodes[idx]['smile'] = smile
                    g.nodes[idx]['jtnode'] = node_y 
                    g.nodes[idx]['smarts'] = smarts

                    bfs_queue.append(node_left)

                if node_right is not None and not CNJMolUtil.is_dummy(node_right.data['smile']):                     
                    if node_x  in parent_map:
                        node_x = parent_map[node_x]

                    idx = node_right.idx
                    vocid = node_right.data['idx_voc']
                    smile = node_right.data['smile']
                    smarts = node_right.data['smarts']
                    wid = ctoken.STDTokens.vocab.get_index(smile)  

                    slots = MolTreeUtils.get_slots(smile)
                    fa_slot = MolTreeUtils.get_slots(node_x.smiles)

                    node_y = MolTreeNode(smile)

                    next_wid = wid
                    next_slots = slots

                    node_y.wid = next_wid
                    node_y.idx = idx
                    node_y.neighbors.append(node_x)
                    node_x.neighbors.append(node_y)

                    all_nodes.append(node_y)

                    stack.append((node_y, next_slots))
                    id_map[idx] = (node_y, next_slots)                    
                    parent_map[node_y] = node_x
               
                    g.add_edge(node_x.idx, node_y.idx)
                    g.nodes[idx]['nid'] = idx
                    g.nodes[idx]['smile'] = smile
                    g.nodes[idx]['jtnode'] = node_y 
                    g.nodes[idx]['smarts'] = smarts
               
                    bfs_queue.append(node_right)

        except Exception as e:
            print(e.args)
       

        n_nodes = len(g.nodes)
        moltree.nodes = []

        for idkey in g.nodes:
            node = g.nodes[idkey]
            if CNJMolUtil.is_dummy(node['smile']):
                continue
            else:
                moltree.nodes.append(node['jtnode'])

        for i, node in enumerate(moltree.nodes):
            node.idx = i
            node.nid = node.idx + 1
            node.is_leaf = (len(node.neighbors) == 1)
            node.candidates = []

        dummy_nodes = []
        for node in moltree.nodes: 
            if CNJMolUtil.is_dummy(node.smiles):
                node.neighbors[0].neighbors.remove(node)
                dummy_nodes.append(node)
                continue

            if node.mol is None:
                node.smiles = CNJMolUtil.valid_smiles(node.smiles, ctoken = ctoken)
                node.mol = Chem.MolFromSmiles(node.smiles)
            
            node.is_leaf = (len(node.neighbors) == 1)

        moltree.graph_nx = g

        return root, all_nodes, g, moltree

    def get_target_candidates(candidates, n_candidates, alg = 'random'):
        if len(candidates) < n_candidates:
            cands = candidates
        else:
            print(f'[len of candidates]:{len(candidates)}, select the first {n_candidates} as target')

            cands = []
            
            if alg == 'atomenv':
                cand = candidates[0]                 
                if len(cand) < 4:
                    alg = 'random'
                else:      
                    scores = []
                    for cand in candidates:
                        scores.append(cand[3])
                    scores = sorted(scores)

                    cands = sorted(candidates, reverse=True, key = lambda x: x[3])
                    cands = cands[:n_candidates]
            elif alg.startswith('goal_'):  #'goal_valsartan_smarts'
                flag = True
                gfun_name = alg[5:]

                module_name = 'External.Guacamol.guacamol.standard_benchmarks'               

                gfun = ModelUtils.str_to_class(module_name = module_name,  class_name = gfun_name)
                if gfun is not None:
                    gfun = gfun()

                    sml_cands = []
                    for cand in candidates:
                        sml_cands.append(cand[0])

                    scores =  gfun.objective.score_list(sml_cands)
                    if sum(scores) > 0:
                        p_list = []
                        for i, cand in enumerate(candidates):
                            candidates[i] = list(candidates[i])
                            candidates[i][3] = scores[i]
                    
                        cands = sorted(candidates, reverse=True, key = lambda x: x[3])
                        cands = cands[:n_candidates]
                        flag = False

                if flag:
                    index = list(range(0, len(candidates) - 1))
                    random.shuffle(index)

                    i = 0;
                    while i < n_candidates:
                        cands.append(candidates[index[i]])
                        i += 1           
            else: # alg == 'random':
                index = list(range(0, len(candidates) - 1))
                random.shuffle(index)

                i = 0;
                while i < n_candidates:
                    cands.append(candidates[index[i]])
                    i += 1             

        return cands

    def assemble_JTMolTree_leaf_first(moltree,
                                      uvisited,
                                      hparam = CNJASM_HParam(),
                                     ):
        print_info  = False

        n_nodes = len(uvisited)
        node_leaf = []
        dec_sml = 'CC'

        for node in moltree.nodes:    
            node.is_leaf = (len(node.neighbors) == 1)

        while(sum(uvisited) > 1):   
            node_leaf = []
            for i, node in enumerate(moltree.nodes):
                if node.is_leaf and uvisited[node.idx] == 1:
                    node_leaf.insert(0,node)  
        
            for node in node_leaf:
                neib = node.neighbors  
                neib_cand = []
                if len(neib) > 0:
                    gened = []

                    candidates = []
                    if hparam['candidate_mode'] == 'random_one':
                        candidates = CNJMolAssembler.enum_assemble(node, neib, prev_nodes = [], prev_amap = [], alg = alg)
                    else:  #candidate_tree
                        if len(node.candidates) == 0:
                            candidates = CNJMolAssembler.enum_assemble(node, neib, prev_nodes = [], prev_amap = [], hparam = hparam)
                        else:
                            can_smls = []
                            for cand in node.candidates:  
                                node_copy = copy.deepcopy(node)
                                node_copy.smiles = cand[0]

                                if len(neib[0].candidates) == 0:
                                    cands = CNJMolAssembler.enum_assemble(node_copy, neib, prev_nodes = [], prev_amap = [], hparam = hparam)
                                    for cd in cands:
                                        if cd[0] not in can_smls:
                                            can_smls.append(cd[0])
                                            candidates.append(cd)
                                else:
                                    old_candidates = neib[0].candidates

                                    for i, bn_cand in enumerate(neib[0].candidates):
                                        nb_copy = copy.deepcopy(neib[0])
                                        nb_copy.smiles = bn_cand[0]

                                        cands = CNJMolAssembler.enum_assemble(node_copy, [nb_copy], prev_nodes = [], prev_amap = [], hparam = hparam)
                                        for cd in cands:
                                            if cd[0] not in can_smls:
                                                can_smls.append(cd[0])
                                                candidates.append(cd)

                                        if len(cands) > 0:
                                            gened.append((bn_cand, cands))    

                    n_candidates = hparam['n_candidates']  #25
                    if len(candidates) > n_candidates:
                        candidates = CNJMolAssembler.get_target_candidates(candidates, n_candidates, 
                                                                            alg = hparam['cand_alg'] #'random','atomenv',
                                                                            )   
                    elif len(candidates) > 1 :
                        candidates = sorted(candidates, reverse=True, key = lambda x: x[3])                   

                    neib[0].candidates.extend(candidates)

                    tgt_smls = RDKFragUtil.verify_candidates(neib[0].candidates)  
                    neib[0].candidates = []
                    for sml in tgt_smls:
                        neib[0].candidates.append((sml, Chem.MolFromSmiles(sml)))        

                    if len(neib[0].candidates) > 0: 
                        idx = 0
                        if hparam['candidate_mode'] == 'random_one':
                            idx = random.randint(0,len(neib[0].candidates)-1)  

                        neib[0].smiles = candidates[idx][0]
                        neib[0].mol = candidates[idx][1]

                    uvisited[node.idx] = 0
                    neib[0].neighbors.remove(node)               
                break 

            for node in moltree.nodes:
                node.is_leaf = (len(node.neighbors) == 1)

        #end while

        idx = uvisited.index(1)
        final = moltree.nodes[idx] 

        if len(final.candidates) == 0: 
            final.candidates = [(final.smiles, Chem.MolFromSmiles(final.smiles))]

        return final

    def can_be_assembled(node):
        neighbors = node.neighbors[0]

        n_degree = 0
        for nb in neighbors.neighbors:
            if nb.degree() > 2:
                return False

            if nb.degree() > 1:
                n_degree += 1

        if n_degree > 1:
            return False
        else:
            return True

    def assemble_one_node(node, 
                          uvisited, 
                          hparam = CNJASM_HParam(),
                          ):
        try:
            neibs = node.neighbors

            if hparam['candidate_mode'] == 'random_one':
                candidates = CNJMolAssembler.enum_assemble(node, neibs, prev_nodes = [], prev_amap = [], hparam = hparam)
                if len(candidates) > hparam['n_candidates']:
                    candidates = CNJMolAssembler.get_target_candidates(candidates, n_candidates = hparam['n_candidates'], 
                                                                        alg = hparam['can_alg'] #'random','atomenv',
                                                                        )  
            else:  
                if len(node.candidates) == 0:
                    candidates = CNJMolAssembler.enum_assemble(node, neibs, prev_nodes = [], prev_amap = [], hparam = hparam)
                else:
                    candidates = []
                    for cand_sml in node.candidates:  
                        node_copy = copy.deepcopy(node)
                        node_copy.smiles = cand_sml

                        cands = CNJMolAssembler.enum_assemble(node_copy, neibs, prev_nodes = [], prev_amap = [], hparam = hparam)
                        candidates.extend(cands)

            if len(candidates) == 0:
                print('assemble_one_node:Candidates is None! Try agin')
                try:
                    i = 1
                    while len(candidates) == 0 and i < len(neibs):
                        candidates = CNJMolAssembler.enum_assemble(node, neibs[0 : -i], prev_nodes = [], prev_amap = [], hparam = hparam)
                        i = i + 1

                except Exception as e:
                    print('assemble_one_node.enum_assemble again Exception!')
                    print(e.args)


            if len(candidates) > 10:
                index = list(range(0, len(candidates) - 1))
                random.shuffle(index)

                cands = []
                i = 0;
                while i < 10:
                    cands.append(candidates[index[i]])
                    i = i + 1

                candidates = cands

            #----------
            for nb in neibs:
                uvisited[nb.idx] = 0
                nb.neighbors.remove(node)

            leave_nb = []  
            for nb in neibs:
                if nb.degree() > 0:
                    leave_nb.append(nb)

            if len(leave_nb) > 1:
                raise ValueError('assemble_JTMolTree_degree_first get more node which input degree is more than one!')        
            else:
                idx = -1
                if len(candidates) > 0:  
                    idx = random.randint(0,len(candidates)-1) 

                if idx >=0:
                    if len(leave_nb) == 0:
                        node.smiles = candidates[idx][0]
                        node.mol = candidates[idx][1]        
                        node.candidates.extend(candidates)
                    else:
                        leave_nb[0].smiles = candidates[idx][0]
                        leave_nb[0].mol = candidates[idx][1]       
                        leave_nb[0].candidates.extend(candidates)
                        uvisited[leave_nb[0].idx] = 1
                
                        uvisited[node.idx] = 0
                        node.neighbors.clear()     
                else:
                    print('assemble_one_node:Candidates is None! Ignore assemble info, keep node smiles')
                    if len(leave_nb) == 1:
                        uvisited[leave_nb[0].idx] = 1
                        uvisited[node.idx] = 0
                        node.neighbors.clear() 
                    
        except Exception as e:
            print('assemble_one_node.Exception!')
            print(e.args)

        return


    def assemble_JTMolTree_degree_first(moltree,
                                      uvisited, 
                                      hparam = CNJASM_HParam()
                                      ):
        n_nodes = len(uvisited)
        node_leaf = []
        dec_sml = 'CC'
        max_degree = 1
        maxd_node = None

        for node in moltree.nodes:
            node.is_leaf = (len(node.neighbors) == 1)

        try:
            while(sum(uvisited) > 1):   
                node_leaf = []
                for i, node in enumerate(moltree.nodes):
                    if node.degree() == 1 and uvisited[node.idx] == 1:
                        node_leaf.append(node) 

                no_assemble = True

                for node in node_leaf:
                    if CNJMolAssembler.can_be_assembled(node):
                        CNJMolAssembler.assemble_one_node(node.neighbors[0], uvisited , hparam)

                        no_assemble = False
                        break

                if no_assemble:
                    node  = node_leaf[0]
                    CNJMolAssembler.assemble_one_node(node, uvisited, hparam)
            
                for node in moltree.nodes:
                    node.is_leaf = (len(node.neighbors) == 1)

            idx = uvisited.index(1)
            final = moltree.nodes[idx]
        except Exception as e:
            print('assemble_JTMolTree_degree_first.Exception:')
            print(e.args)

        return final

    def assemble_order(moltree, use_stereo = False):
        pred_root = moltree.nodes[0]
        pred_nodes = moltree.nodes

        for i, node in enumerate(pred_nodes):
            node.nid = i + 1  
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                ChemUtils.set_atommap(node.mol, node.nid)

        tree_mess = None  
        mol_vec = None

        cur_mol = ChemUtils.copy_edit_mol(pred_root.mol)

        global_amap = [{}] + [{} for node in pred_nodes]  
        global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in
                          cur_mol.GetAtoms()} 

        cur_mol = CNJMolAssembler.dfs_assemble_without_score(tree_mess      = tree_mess, 
                                                             mol_vec        = mol_vec, 
                                                             all_nodes      = pred_nodes, 
                                                             cur_mol        = cur_mol,
                                                             global_amap    = global_amap, 
                                                             fa_amap        = [], 
                                                             cur_node       = pred_root, 
                                                             fa_node        = None,
                                                             n_random       = 5,
                                                             )
        final = cur_mol

        if cur_mol is not None:
            cur_mol = cur_mol.GetMol()
            ChemUtils.set_atommap(cur_mol)
            final = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))

        return final

    def dfs_assemble_without_score(tree_mess, mol_vec, all_nodes, cur_mol,
                                   global_amap, fa_amap, 
                                   cur_node, 
                                   fa_node,  #previous node, it mean which is father
                                   n_random = 5,
                                   ):
        fa_nid = fa_node.nid if fa_node is not None else -1  
        prev_nodes = [fa_node] if fa_node is not None else []  

        children = [nei for nei in cur_node.neighbors if  nei.nid != fa_nid]  
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]  
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]  
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid, a2, a1) for nid, a1, a2 in fa_amap if nid == cur_node.nid]  

        cands = ChemUtils.enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)  
 
        #n_random = 5
        if len(cands) > n_random:
            index = list(range(0, len(cands) - 1))
            random.shuffle(index)

            candidates = []
            i = 0;
            while i < n_random:
                candidates.append(cands[index[i]])
                i += 1

            cands = candidates

        if len(cands) == 0:
            return None

        cand_smiles, cand_mols, cand_amap = zip(*cands)

        cands = [(candmol, all_nodes, cur_node) for candmol in cand_mols]


        if len(cands) == 1:
            cand_idx = [0]
        else:
            cand_idx = [*range(len(cands) - 1, 0, -1)]

        backup_mol = Chem.RWMol(cur_mol)
        for i in range(len(cand_idx)):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[int(cand_idx[i])]  
            new_global_amap = copy.deepcopy(global_amap)  

            for nei_id, ctr_atom, nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom] 

            cur_mol = ChemUtils.attach_mols(cur_mol, children, [], new_global_amap)  
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))


            if new_mol is None:
                continue

            result = True
            for nei_node in children:
                if nei_node.is_leaf:
                    continue
                cur_mol_b = CNJMolAssembler.dfs_assemble_without_score(tree_mess      = tree_mess, 
                                                                     mol_vec        = mol_vec, 
                                                                     all_nodes      = all_nodes, 
                                                                     cur_mol        = cur_mol,
                                                                     global_amap    = new_global_amap,
                                                                     fa_amap        = pred_amap, 
                                                                     cur_node       = nei_node, 
                                                                     fa_node        = cur_node,
                                                                     n_random       = n_random,
                                                                     )
                if cur_mol_b is None:
                    result = False
                    break
                else:
                    cur_mol = cur_mol_b

            if result:
                return cur_mol

        return cur_mol


    def assemble_JTMolTree(moltree: CNJTMolTree, 
                           ctoken = None,
                           hparam = CNJASM_HParam(),
                           ):
        n_nodes = len(moltree.nodes)
        uvisited = [1] * n_nodes 
        node_leaf = []
        dec_sml = 'CC'

        try:
            for node in moltree.nodes:
                if node.smiles == '&':
                    uvisited[node.idx] = 0
                    neib = node.neighbors[0].neighbors.remove(node)
                else:
                    node.smiles = CNJMolUtil.valid_smiles(node.smiles, ctoken = ctoken)

            for node in moltree.nodes:
                node.is_leaf = (len(node.neighbors) == 1)

            if hparam['alg'] == 'CALD_TSSA':
                for node in moltree.nodes:  
                    if len(node.neighbors) > 1:
                        ChemUtils.set_atommap(node.mol, node.nid)

                if hparam['assemble_alg'] == 'JTVAE_decode':
                    if hparam['n_candidates'] > 1:  #get more candidates for Goal-directed Reconstruction for TSSA,, set ncand < 20 in ChemUtils.enum_assemble()
                        final = []
                        for i in range(hparam['n_candidates']):
                            final.append(CNJMolAssembler.assemble_order(moltree))
                    else:
                        final = CNJMolAssembler.assemble_order(moltree)

                    if final is None:
                        final = CNJMolAssembler.assemble_JTMolTree_degree_first(moltree, uvisited = uvisited, hparam = CNJMolAssembler.get_hparam(asm_alg = 'CALG_TSDY'))  
                    #CNJMolAssembler.assemble_jtvae_decode(moltree)
                elif hparam['assemble_alg'] == 'group_first':
                    final = CNJMolAssembler.assemble_JTMolTree_degree_first(moltree, uvisited = uvisited, hparam = CNJMolAssembler.get_hparam(asm_alg = 'CALG_TSDY'))           
                else:
                    final = CNJMolAssembler.assemble_JTMolTree_leaf_first(moltree,uvisited = uvisited,  hparam = CNJMolAssembler.get_hparam(asm_alg = 'CALG_TSDY'))

                cur_mol =  CNJMolAssembler.get_final_mol(final, 
                                                         final_select_alg   = hparam['final_select_alg'], 
                                                         use_stereo         = hparam['use_stereo'],
                                                         p_mean             = hparam['pMean'],
                                                         ) 
                
            elif hparam['alg'] == 'CALG_TSDY' or hparam['alg'] == 'CALG_TSID' or hparam['alg'] == 'CALG_TSIS':
                final = CNJMolAssembler.assemble_JTMolTree_leaf_first(moltree, uvisited = uvisited, hparam = hparam)

                cur_mol =  CNJMolAssembler.get_final_mol(final, 
                                                         final_select_alg   = hparam['final_select_alg'], 
                                                         use_stereo         = hparam['use_stereo'],
                                                         p_mean             = hparam['pMean'],
                                                         longest_mol        = True,
                                                         )   
            else: #alg == 'BRICS':
                cur_mol = CNJMolAssembler.assemble_Brics_bfs(moltree, uvisited = uvisited,)
                cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))              

            try:
                if isinstance(cur_mol, rdkit.Chem.rdchem.Mol):
                    s_mol = RDKUtils.remove_dummy_atom(cur_mol)
                    #Chem.SanitizeMol(s_mol)  #added at 2023.09.04

                    dec_sml = Chem.MolToSmiles(s_mol)
                    dec_sml = RDKUtils.remove_atommap_info(dec_sml)

                    #new_mol = RDKUtils.remove_atom(Chem.MolFromSmiles(dec_sml), atomic_num = 0)
                    #Chem.Kekulize(Chem.MolToSmiles(dec_sml))
                    #dec_sml = Chem.MolToSmiles(new_mol, kekuleSmiles=True)
                    #dec_sml = Chem.MolToSmiles(new_mol)
                else:
                    dec_sml_list = []
                    for s_mol in cur_mol:
                        s_mol = RDKUtils.remove_dummy_atom(s_mol)
                        #Chem.SanitizeMol(s_mol)  #added at 2023.09.04

                        dec_sml = Chem.MolToSmiles(s_mol)
                        dec_sml = RDKUtils.remove_atommap_info(dec_sml)

                        #new_mol = RDKUtils.remove_atom(Chem.MolFromSmiles(dec_sml), atomic_num = 0)
                        #Chem.Kekulize(Chem.MolToSmiles(dec_sml))
                        #dec_sml = Chem.MolToSmiles(new_mol, kekuleSmiles=True)
                        #dec_sml = Chem.MolToSmiles(new_mol)

                        dec_sml_list.append(dec_sml)

                    dec_sml_list.sort()
                    dec_sml = dec_sml_list

            except Exception as e:
                print('[[CNJMolAssembler.assemble_JTMolTree.select final].Exception',e.args)
                if isinstance(cur_mol, rdkit.Chem.rdchem.Mol):
                    dec_sml = Chem.MolToSmiles(cur_mol)
                else:
                    dec_sml = Chem.MolToSmiles(cur_mol[0])

        except Exception as e:
            print('[CNJMolAssembler.assemble_JTMolTree]:', e.args)

        return dec_sml


    def get_final_mol(final_node, final_select_alg = 'random', use_stereo = True, p_mean = None, longest_mol = False):
        if isinstance(final_node, rdkit.Chem.rdchem.Mol):
            if use_stereo:
                isomers = list(EnumerateStereoisomers(final_node))
                final_node = random.choice(isomers)
            #Chem.SanitizeMol(final_mol)  #???
            return final_node
        
        try:
            sml_cands = list(set([cand[0] for cand in final_node.candidates]))
        except Exception as e:
            sml_cands = list(set([Chem.MolToSmiles(cand) for cand in final_node]))


        final_mol = None        
        
        try:
            id_cands = []

            if final_select_alg == 'random':
                sml = random.choice(sml_cands)
                final_mol = Chem.MolFromSmiles(sml)

            elif final_select_alg == 'candidates':
                mol_list = []
                for i, sml in enumerate(sml_cands):
                    mol = Chem.MolFromSmiles(sml)
                    if mol is None:
                        continue
                    mol_list.append(mol)

                    if use_stereo:
                        isomers = list(EnumerateStereoisomers(mol))
                        mol_list.extend(isomers)

                mol_list = list(set(mol_list))

                final_mol = mol_list
               
                if longest_mol:
                    cands_smls, final_mol =  RDKFragUtil.get_longest_mols(final_mol)
            elif final_select_alg.startswith('goal_'):  #'goal_valsartan_smarts'
                flag = True
                gfun_name = final_select_alg[5:]

                #External.Guacamol.guacamol.goal_directed_benchmark.GoalDirectedBenchmark
                module_name = 'External.Guacamol.guacamol.standard_benchmarks'               

                gfun = ModelUtils.str_to_class(module_name = module_name,  class_name = gfun_name)
                if gfun is not None:
                    gfun = gfun()
                    scores =  gfun.objective.score_list(sml_cands)

                    if sum(scores) > 0:
                        p_list = []
                        for i, item in enumerate(sml_cands):
                            p_list.append((scores[i], sml_cands[i]))
                    
                        p_list.sort(key=lambda u:(-u[0]))
                        final_mol = Chem.MolFromSmiles(p_list[0][1])   
                        flag = False

                if flag:                   
                    sml = random.choice(sml_cands)
                    final_mol = Chem.MolFromSmiles(sml)
            else:
                p_list = []
                mol_list = []
                for i, sml in enumerate(sml_cands):
                    mol = Chem.MolFromSmiles(sml)
                    if mol is None:
                        continue
                    mol_list.append(mol)

                    if use_stereo:
                        isomers = list(EnumerateStereoisomers(mol))
                        mol_list.extend(isomers)
                #end for

                for mol in mol_list:
                    p = CNJMolAssembler.get_mol_properties(mol)
                    p_list.append([p, mol])

                if p_mean is not None:
                    for i in range(len(p_list)):
                        p_list[i] = np.abs(p_list[i][0] - p_mean)

                #p_list.sort(key=lambda u:(u[0], -u[1]))
                p_list.sort(key=lambda u:(-u[0]))

                final_mol = p_list[0][1]
        except Exception as e:
            print('[CNJMolAssembler.get_final_mol].Exception', e.args)
            final_mol = Chem.MolFromSmiles('CC')
 
        if final_mol is None:
            final_mol = [Chem.MolFromSmiles('CC')]
        elif isinstance(final_mol, rdkit.Chem.rdchem.Mol):
            #final_mol = RDKUtils.remove_dummy_atom(final_mol)
            dec_sml =  Chem.MolToSmiles(final_mol)
            #dec_sml = RDKUtils.remove_atommap_info(dec_sml)

            fix_smls = RDKFragUtil.fix_mol(dec_sml)
            if fix_smls is not None:
                final_mol = Chem.MolFromSmiles(fix_smls)
            else:
                final_mol = [Chem.MolFromSmiles('CC')]
            final_mol = [final_mol]
        else:
            for i, join_mol in enumerate(final_mol):
                joined_sml =  Chem.MolToSmiles(join_mol)
                fix_smls = RDKFragUtil.fix_mol(joined_sml)
                if fix_smls is not None:
                    final_mol[i] = Chem.MolFromSmiles(fix_smls)        

        return final_mol 


    def dfs_assemble(all_nodes, cur_mol, global_amap, fa_amap, cur_node, fa_node, prob_decode):
        fa_nid = fa_node.nid if fa_node is not None else -1  
        prev_nodes = [fa_node] if fa_node is not None else []  

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]     
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]     
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]    

        neighbors = singletons + neighbors

        cur_amap = [(fa_nid, a2, a1) for nid, a1, a2 in fa_amap if nid == cur_node.nid] 

        cands = CNJMolAssembler.enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)  
        if len(cands) == 0:
            return None

        cand_smiles, cand_mols, cand_amap = zip(*cands)
        cands = [(candmol, all_nodes, cur_node) for candmol in cand_mols]

        cand_idx = [0]

        backup_mol = Chem.RWMol(cur_mol)
        for i in range(len(cand_idx)):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i]]  
            new_global_amap = copy.deepcopy(global_amap) 

            for nei_id, ctr_atom, nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]  

            cur_mol = ChemUtils.attach_mols(cur_mol, children, [], new_global_amap) 
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))
  
            if new_mol is None: continue

            result = True
            for nei_node in children:
                if nei_node.is_leaf: continue
                cur_mol = CNJMolAssembler.dfs_assemble(all_nodes, 
                                                       cur_mol, 
                                                       new_global_amap,
                                                       pred_amap,
                                                       nei_node, 
                                                       cur_node, 
                                                       prob_decode
                                                       )
                if cur_mol is None:
                    result = False
                    break

            if result:
                return cur_mol
        return None

    def Assemble_Mol(vocab,
                    ctoken,
                    bfs_binary_tree:CNJTMolTreeNode,
                    bfs_node_map,
                    g,  # a amt graph or a CNXBinaryTree, 
                        #if it is a amt graph, bfs_binary_tree is not linked
                        #if CNXBinaryTree, bfs_binary_tree is linked
                    hparam = CNJASM_HParam(),
                    show = True,
                    ):
        dec_smile = ''
        try:
            if hparam['alg'] == 'CALG_TSIS':
                pred_root, pred_nodes, g, moltree = CNJMolAssembler.Convert_amt_JTMoltree(vocab, ctoken, g, bfs_node_map, show)
            else:
                pred_root, pred_nodes, g, moltree = CNJMolAssembler.Convert_to_JTMoltree(vocab, ctoken, bfs_binary_tree, bfs_node_map, show)


            dec_smile = CNJMolAssembler.assemble_JTMolTree(moltree, 
                                                           ctoken           = ctoken,
                                                           hparam           = hparam,
                                                           )
        except Exception as e:
            print(e.args)
            
        if dec_smile is not None:
            if isinstance(dec_smile, str):
                if len(dec_smile) == 0:
                    dec_smile = 'CC'
            else:
                for i, sml in enumerate(dec_smile):
                    if len(sml) == 0:
                        dec_smile[i] = 'CC'         
                if len(dec_smile) == 1:
                    dec_smile  = dec_smile[0]         
        else:
            dec_smile = 'CC' 

        return dec_smile


    def decode_single(bfs_ex_smiles, ctoken, asm_alg = 'CALG_TSDY', n_samples = 1, p_mean = None):
        try:                                                       
            hparam = CNJMolAssembler.get_hparam(asm_alg)
 
            sub_smils = bfs_ex_smiles.strip().split('.')  
            re_smils = ''
            errors = []
            re_ex_smils = []
            new_vocs = []

            for i, sub_s in enumerate(sub_smils):
                if not isinstance(sub_s, (str)):
                    print(f'[test_rebuild_file is not a string-{i}:-{bfs_ex_smiles}]')
                    continue
                else:
                    bfs_ex_smiles = CNJMolUtil.split_ex_smiles(sub_s, delimiter='^')

                if i > 0:
                    re_smils += '.'
                    re_ex_smils+= '.'

                if bfs_ex_smiles is None or len(bfs_ex_smiles) < 1:
                    print(f'[test_rebuild_file is null or empty-{i}:-{bfs_ex_smiles}]')
                    continue

                if len(bfs_ex_smiles) > ctoken.max_length:
                    bfs_ex_smiles = bfs_ex_smiles[0:ctoken.max_length]

                skt_wrong = 0
                if hparam['alg'] == 'CALG_TSIS':
                    bfs_tree, g_amt, bfs_node_map =  CNJTMolTree.bfs_ex_reconstruct_amt(ctoken = ctoken, bfs_ex_smiles = bfs_ex_smiles, show = False) # bfs_tree is not linked
                else:
                    g_amt = None
                    bfs_tree, g_CNXBinaryTree, bfs_node_map, skt_wrong =  CNJTMolTree.bfs_ex_reconstruct(ctoken = ctoken, bfs_ex_smiles = bfs_ex_smiles, show = False)# bfs_tree is linked
                
                if bfs_tree is not None:                      
                    bfs_ex_nodeid, bfs_ex_vocids, bfs_ex_smiles, new_voc, bfs_ex_smarts= CNJTMolTree.get_bfs_ex(ctoken, bfs_tree)  #generate advanced bfs 
                    
                    if hparam['alg'] == 'CALG_TSSA' or hparam['alg'] == 'CALG_TSDY' or hparam['alg'] == 'CALG_TSID':
                        if len(bfs_ex_smiles) % 2 == 1:
                            bfs_ex_smiles.append['&']

                        if bfs_ex_smiles is not None:
                            re_ex_smils += bfs_ex_smiles

                    if len(new_voc)> 0:
                        new_vocs.append(new_voc)

                    for k in range(n_samples):
                        #Dummy: alg = 'CALG_TSDY',  assemble_alg = 'leaf_first'
                        #other: alg = 'CALG_TSSA'   assemble_alg = 'JTVAE_decode'

                        dec_smile =  CNJMolAssembler.Assemble_Mol(vocab             = None,    #vocab,
                                                                    ctoken          = ctoken,
                                                                    bfs_binary_tree = bfs_tree,
                                                                    bfs_node_map    = bfs_node_map,
                                                                    g               = g_amt,
                                                                    show            = False,
                                                                    hparam          = hparam,
                                                                    )


                        if not isinstance(dec_smile, str):
                            dec_smile = '|!|'.join(s for s in dec_smile)

                        if dec_smile is not None and len(dec_smile)>0:
                            re_smils += dec_smile
                        else:
                            errors.append(bfs_ex_smiles)    

        except Exception as e:
            print('[CNJMolAssembler.decode_single].Exception:', e.args)
            print('[CNJMolAssembler.decode_single].Exception:', bfs_ex_smiles)
            re_smils = 'CC'
            bfs_ex_smiles = ['CC', '&', '&', '&']
            new_vocs = [['CC']]

        return re_smils, bfs_ex_smiles, new_vocs, skt_wrong



def rebuild_file(n_samples = 1):
    maxlen = 512

    #vocab_file = r'../RawData/Chembl/Test/Chembl_test.smi.[JTVAE][131]_token.voc.smi'
    #vocab_file = r'../RawData/Chembl/Test/Chembl_test.smi.[BRICS_Base][94]_token.voc.smi'
    #vocab_file = r'../RawData/Chembl/Test/Chembl_test.smi.[MMPA][125]_token.voc.smi'
    #vocab_file = r'../RawData/Chembl/Test/Chembl_test.smi.[Scaffold][103]_token.voc.smi'

    #vocab_file = r'../RawData/Chembl/Test/Chembl_test.smi.[BRICS_DY][102]_token.voc.smi'
    #vocab_file = r'../RawData/Chembl/Test/Chembl_test.smi.[MMPA_DY][237]_token.voc.smi'
    #vocab_file = r'../RawData/Chembl/Test/Chembl_test.smi.[Scaffold_DY][148]_token.voc.smi'
    #vocab_file = r'H:\RawData\ChEMBL\Test\Chembl_test.smi.[Scaffold_DY][148]_token.voc.smi'

    #vocab_file = r'../RawData/Chembl/Test/Chembl_test.smi.[Vanilla][94]_token.voc.smi'
      
    #vocab_file = r'D:\ProjectTF\RawData\examples\WT\mol.smi.[BRICS_DY][142]_token.voc.smi'
    #vocab_file = r'D:\ProjectTF\RawData\examples\WT\mol.smi.[MMPA_DY][188]_token.voc.smi'
    #vocab_file = r'D:\ProjectTF\RawData\examples\WT\mol.smi.[Scaffold_DY][160]_token.voc.smi'

    vocab_file = None

    ctoken = CTokens(STDTokens_Frag_File(vocab_file), is_pad = True, pad_symbol = ' ', startend = True,
                     max_length = maxlen,  flip = False, invalid = True, onehot = False)    
    
    #TSSA
    #smlfile = r'../RawData/Chembl/Test/Chembl_test.smi.[JTVAE][131]_TSSA.csv'
    #smlfile = r'../RawData/Chembl/Test/Chembl_test.smi.[BRICS_Base][94]_TSSA.csv'
    #smlfile = r'../RawData/Chembl/Test/Chembl_test.smi.[MMPA][125]_TSSA.csv'
    #smlfile = r'../RawData/Chembl/Test/Chembl_test.smi.[Scaffold][103]_TSSA.csv'
     
    ##TSDY
    #smlfile = r'../RawData/Chembl/Test/Chembl_test.smi.[BRICS_DY][102]_TSDY.csv'
    #smlfile = r'../RawData/Chembl/Test/Chembl_test.smi.[MMPA_DY][237]_TSDY.csv'
    #smlfile = r'../RawData/Chembl/Test/Chembl_test.smi.[Scaffold_DY][148]_TSDY.csv'
    
    ##TSID
    #smlfile = r'../RawData/Chembl/Test/Chembl_test.smi.[BRICS_DY][102]_TSID.csv'
    #smlfile = r'../RawData/Chembl/Test/Chembl_test.smi.[MMPA_DY][237]_TSID.csv'
    #smlfile = r'../RawData/Chembl/Test/Chembl_test.smi.[Scaffold_DY][148]_TSID.csv'
    #smlfile = r'H:\RawData\ChEMBL\Test\Chembl_test.smi.[Scaffold_DY][148]_TSID.csv.aug[2].csv'

    ##TS_Vanilla
    #smlfile = r'../RawData/Chembl/Test/Chembl_test.smi.[Vanilla][94]_TSV.csv'
        
    #-------test------------------
    #smlfile = r'D:\ProjectTF\RawData\examples\WT\mol.smi.[BRICS_DY][142]_TSDY.csv'
    #smlfile = r'D:\ProjectTF\RawData\examples\WT\mol.smi.[MMPA_DY][188]_TSDY.csv'
    #smlfile = r'D:\ProjectTF\RawData\examples\WT\mol.smi.[Scaffold_DY][160]_TSDY.csv'

    #smlfile = r'D:\ProjectTF\RawData\examples\WT\mol.smi.[BRICS_DY][142]_TSID.csv'
    #smlfile = r'D:\ProjectTF\RawData\examples\WT\mol.smi.[MMPA_DY][188]_TSID.csv'
    #smlfile = r'D:\ProjectTF\RawData\examples\WT\mol.smi.[Scaffold_DY][160]_TSID.csv'
    
    #----xiaozhi---------------------------------
    #smlfile = r'G:\Report\Xiaozhi\gen_smiles_tsmi_epoch12.csv'

    #-----------------------------

    #smlfile= r'../RawData/Example/mol.smi.[MMPA][103]_TSSA.csv'
    #smlfile= r'../RawData/Example/mol.smi.[MMPA_DY][202]_TSDY.csv'
    #smlfile= r'../RawData/Example/mol.smi.[MMPA_DY][202]_TSID.csv'
    #smlfile= r'../RawData/Example/mol.smi.[MMPA_DY][202]_TSIS.csv'
    #smlfile= r'../RawData/Example/mol.smi.[MMPA_DY][202]_TSISO.csv'
    #smlfile= r'../RawData/Example/mol.smi.[MMPA_DY][202]_TSISR.csv'
    smlfile= r'../RawData/Example/mol.smi.[MMPA_DY][202]_TSISD.csv'
    #-------------------------------
    print(vocab_file)
    print(smlfile)
 
    #asm_alg = 'CALG_TSSA'  
    #asm_alg = 'CALG_TSDY'  
    #asm_alg = 'CALG_TSID'  
    asm_alg = 'CALG_TSIS'  

    re_ex_smils = []
    re_smils_list = []
    errors = []
    new_vocs = []
    skt_wrong_list = []

    skip_blank_lines = True

    df = pd.read_csv(smlfile, squeeze=True, delimiter=',',header = None ,skip_blank_lines = skip_blank_lines)       
    smiles_list = list(df.values)
    try:
        for i, s in enumerate(smiles_list):
            if str(s) != 'nan': 
                smiles_list[i] = ''.join(s.strip().split(' '))
            else:
                smiles_list[i] = 'C'
    except Exception as e:
        print('Exception:', e.args)
        return


    for i, bfs_ex_smiles in tqdm(enumerate(smiles_list), total = len(smiles_list),  desc = 'parsing smiles ...'):
        for k in range(n_samples):
            re_smils, bfs_ex_smiles_sub, new_vocs_sub, skt_wrong = CNJMolAssembler.decode_single(bfs_ex_smiles, ctoken, asm_alg, n_samples = 1, p_mean = None) 
            if re_smils is None or re_smils == '':
                errors.append(f'[{i}],{bfs_ex_smiles}')

            re_smils_list.append(re_smils)
            new_vocs.extend(new_vocs_sub)
            skt_wrong_list.append(0 if skt_wrong==0 else 1)
  
    output = smlfile + f'.re_smils[{n_samples}].smi'
    df = pd.DataFrame(re_smils_list)
    df.to_csv(output, index = False, header=False, na_rep="NULL")
    print(output)

    output = smlfile + f'.errors[{n_samples}].csv'
    df = pd.DataFrame(errors)
    df.to_csv(output, index = False, header=False, na_rep="NULL")
    print(output)
    
    output = smlfile + f'.skt_wrong[{sum(skt_wrong_list)}].csv'
    df = pd.DataFrame(skt_wrong_list)
    df.to_csv(output, index = False, header=False, na_rep="NULL")
    print(output)
   
    new_vocs = [x for l in new_vocs for x in l] 
    new_vocs.sort()
    output = smlfile + f'.new_vocs[{n_samples}].smi'
    df = pd.DataFrame(list(set(new_vocs)))
    df.to_csv(output, index = False, header=False, na_rep="NULL")      
    print(output)

    return


def test_decode():
    
    #-----test skt_wrong------------------------------------

    #-------Celecoxib--------------------------------
    TSID_M  = '[1*]C&[1*]C1=CC=C([2*])C=C1&[2*]C1=CC([3*])=NN1[5*]&[3*]C([4*])(F)F&[4*]F^[5*]C1=CC=C([6*])C=C1&& [6*]S(N)(=O)=O&&&'
    TSIS_M  = '[1*]C^[1*]C1=CC=C([2*])C=C1^[2*]C1=CC([3*])=NN1[5*]^[3*]C([4*])(F)F^[5*]C1=CC=C([6*])C=C1^[4*]F^[6*]S(N)(=O)=O'
    TSISD_M = '[1*]C^[1*]C1=CC=C([2*])C=C1^[2*]C1=CC([3*])=NN1[5*]^[3*]C([4*])(F)F^[4*]F^[5*]C1=CC=C([6*])C=C1^[6*]S(N)(=O)=O'
    TSISO_M = '[2*]C1=CC([3*])=NN1[5*]^[1*]C1=CC=C([2*])C=C1^[5*]C1=CC=C([6*])C=C1^[3*]C([4*])(F)F^[6*]S(N)(=O)=O^[1*]C^[4*]F'
    TSISR_M = '[6*]S(N)(=O)=O^[1*]C^[2*]C1=CC([3*])=NN1[5*]^[1*]C1=CC=C([2*])C=C1^[3*]C([4*])(F)F^[5*]C1=CC=C([6*])C=C1^[4*]F'

    #tsmile = TSID_M
    tsmile = TSIS_M
    #tsmile = TSISD_M
    #tsmile = TSISO_M
    #tsmile = TSISR_M
    #----------------------------------------
   


    #asm_alg = CODE_Alg.CALG_TSSA.name   
    #asm_alg = CODE_Alg.CALG_TSDY.name     
    #asm_alg = CODE_Alg.CALG_TSID.name     
    asm_alg = CODE_Alg.CALG_TSIS.name   

    bfs_ex = ''.join(tsmile.strip().split(' '))
    print('input:=', bfs_ex)


    ctoken = CTokens(STDTokens_Frag_File(None))

    #-----------------------------------------

    bfs_ex_smiles = CNJMolUtil.split_ex_smiles(bfs_ex, delimiter='^')
    print('bfs_ex_smiles', bfs_ex_smiles)     
    
    n_samples = 5
    for i in range(n_samples):
        #print('=====[n_samples]===== ',i)
        re_smils, bfs_ex_smiles_sub, new_vocs_sub, skt_wrong = CNJMolAssembler.decode_single(bfs_ex, ctoken , asm_alg, n_samples = 1, p_mean = None) 
        print('dec_smile:=',         re_smils)
        print('bfs_ex_smiles_sub:=', bfs_ex_smiles_sub)
        print('new_vocs_sub:=',      new_vocs_sub)
        print('skt_wrong:=',         skt_wrong)

    return 


if __name__ == '__main__':

    #test_decode()

    rebuild_file()
