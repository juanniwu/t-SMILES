import pandas as pd

from scipy.sparse.construct import random
from tqdm import tqdm

import random
import networkx as nx

import copy

import rdkit
from rdkit import Chem

from DataSet.STDTokens import CTokens, STDTokens_Frag_File

from DataSet.Graph.CNJTMol import CNJTMolTreeNode, CNJTMolTree
from DataSet.Graph.CNJMolUtil import CNJMolUtil
from DataSet.JTNN.MolTree import Vocab, MolTreeNode, MolTreeUtils
from DataSet.JTNN.ChemUtils import ChemUtils

from MolUtils.RDKUtils.Utils import RDKUtils

class CNJMolAssembler:
    def enum_assemble(node, neighbors, prev_nodes=[], prev_amap=[], alg = 'JTVAE'):
        candidates = []
        try:
            if alg ==  'JTVAE':
                candidates = ChemUtils.enum_assemble(node, neighbors, prev_nodes, prev_amap)
            elif alg == 'Brics':
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

            elif alg =='MM':
                candidates = node.smiles
        except Exception as e:
            print('enum_assemble.Exception.node', node.smiles)
            print('enum_assemble.Exception.neighbors', neighbors[0].smiles)
            candidates = []
            print(e.args)

        return candidates

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

        bfs_ex_nodeid, bfs_ex_vocids, bfs_ex_smiles, new_vocs = CNJTMolTree.get_bfs_ex(ctoken, bfs_binary_tree, extra_dummy= False)  #generate advanced bfs

        bfsnode = bfs_binary_tree
        vocid = bfsnode.data['idx_voc']
        smile = bfsnode.data['smile']
        wid = ctoken.STDTokens.vocab.get_index(smile)  #id are different, it should be updated

        root_wid = wid
        root = MolTreeNode(smiles = smile, clique=[])
        root.wid = root_wid
        root.idx = bfsnode.idx
        stack.append((root, ctoken.STDTokens.vocab.get_slots(root.wid)))  # self.vocab.get_slots(720) = [('C', 0, 3), ('C', 0, 3)]      

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
                   jtnode = root
                   )
        try:
            while len(bfs_queue) > 0 and not stop:
                bds_node = bfs_queue.pop(0)
                node_x, fa_slot = id_map[bds_node.idx]

                node_left = bds_node.left
                node_right = bds_node.right

                #------------------------------------
                if node_left is not None and not CNJMolUtil.is_dummy(node_left.data['smile']):
                    idx   = node_left.idx
                    vocid = node_left.data['idx_voc']
                    smile = node_left.data['smile']
                    wid = ctoken.STDTokens.vocab.get_index(smile)  #id are different, it should be updated

                    slots = MolTreeUtils.get_slots(smile)
                    fa_slot = MolTreeUtils.get_slots(node_x.smiles)

                    node_y = MolTreeNode(smile)

                    #if ChemUtils.have_slots(fa_slot, slots) and ChemUtils.can_assemble(node_x, node_y):
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

                    bfs_queue.append(node_left)
                    #else:
                    #    print(f'Convert_to_JTMoltree:node_left[{idx}] could not be assembled!')
                #----------------------------------------
                if node_right is not None and not CNJMolUtil.is_dummy(node_right.data['smile']):                     
                    if node_x  in parent_map:
                        node_x = parent_map[node_x]

                    idx = node_right.idx
                    vocid = node_right.data['idx_voc']
                    smile = node_right.data['smile']
                    wid = ctoken.STDTokens.vocab.get_index(smile)  #id are different, it should be updated

                    slots = MolTreeUtils.get_slots(smile)
                    fa_slot = MolTreeUtils.get_slots(node_x.smiles)

                    node_y = MolTreeNode(smile)

                    #if ChemUtils.have_slots(fa_slot, slots) and ChemUtils.can_assemble(node_x, node_y):
                    next_wid = wid
                    next_slots = slots

                    node_y.wid = next_wid
                    node_y.idx = idx
                    node_y.neighbors.append(node_x)
                    node_x.neighbors.append(node_y)
                    #node_y.clique.append[node_y]

                    all_nodes.append(node_y)

                    stack.append((node_y, next_slots))
                    id_map[idx] = (node_y, next_slots)                    
                    parent_map[node_y] = node_x
               
                    g.add_edge(node_x.idx, node_y.idx)
                    g.nodes[idx]['nid'] = idx
                    g.nodes[idx]['smile'] = smile
                    g.nodes[idx]['jtnode'] = node_y 
               
                    bfs_queue.append(node_right)
                    #else:
                    #    print(f'Convert_to_JTMoltree: node_right[{idx}] could not be assembled!')

        except Exception as e:
            print(e.args)
       
        #if show:          
        #    GTools.show_network_g_cnjtmol(g)
        #-------------------
        n_nodes = len(g.nodes)
        #moltree.nodes = [''] * n_nodes
        moltree.nodes = []

        for idkey in g.nodes:
            node = g.nodes[idkey]
            #moltree.nodes[idkey] = node['jtnode']
            if CNJMolUtil.is_dummy(node['smile']):
                continue
            else:
                moltree.nodes.append(node['jtnode'])

        #update nid to make it same as JTMolNode
        #dummy nodes are removed, so update idx
        for i, node in enumerate(moltree.nodes):
            node.idx = i
            node.nid = node.idx + 1
            node.is_leaf = (len(node.neighbors) == 1)
            node.candidates = []
            #node.candidates = [node.smiles]

        ##
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

    def can_be_assembled(node):
        #node is a leaf
        #node's neighbor's neighbors only include one neighor which degree is 2, all others' degree are 1  
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
                          alg = 'JTVAE',
                          candidate_mode = 'random_one',    #random_one: randomly selece one candidate as target 
                                                                        #candidate_tree: build a candidadate tree
                          ):
        try:
            neibs = node.neighbors #more than one neighbours

            if candidate_mode == 'random_one':
                candidates = CNJMolAssembler.enum_assemble(node, neibs, prev_nodes = [], prev_amap = [], alg = alg)
            else:  #candidate_tree
                #--------------------------------------------                        
                if len(node.candidates) == 0:
                    candidates = CNJMolAssembler.enum_assemble(node, neibs, prev_nodes = [], prev_amap = [], alg = alg)
                else:
                    candidates = []
                    for cand_sml in node.candidates:  #
                        node_copy = copy.deepcopy(node)
                        node_copy.smiles = cand_sml

                        cands = CNJMolAssembler.enum_assemble(node_copy, neibs, prev_nodes = [], prev_amap = [], alg = alg)
                        candidates.extend(cands)

                        #--------------------------------------

            if len(candidates) == 0:
                print('assemble_one_node:Candidates is None! Try agin')
                try:
                    i = 1
                    while len(candidates) == 0 and i < len(neibs):
                        candidates = CNJMolAssembler.enum_assemble(node, neibs[0 : -i], prev_nodes = [], prev_amap = [], alg = alg)
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

            #--------------------------
            for nb in neibs:
                uvisited[nb.idx] = 0
                nb.neighbors.remove(node)

            leave_nb = []  #should only one neighbor left
            for nb in neibs:
                if nb.degree() > 0:
                    leave_nb.append(nb)

            if len(leave_nb) > 1:
                raise ValueError('assemble_JTMolTree_degree_first get more node which input degree is more than one!')        
            else:
                idx = -1
                if len(candidates) > 0:  #select one randomly as target
                    idx = random.randint(0,len(candidates)-1)  #a <= n <= b

                if idx >=0:
                    if len(leave_nb) == 0:
                        #the last one
                        node.smiles = candidates[idx][0]
                        node.mol = candidates[idx][1]        
                        node.candidates.extend(candidates)
                    else:
                        #idx = len(candidates) - 1
                        #idx = 0
                        leave_nb[0].smiles = candidates[idx][0]
                        leave_nb[0].mol = candidates[idx][1]       
                        leave_nb[0].candidates.extend(candidates)
                        uvisited[leave_nb[0].idx] = 1
                
                        #update node
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
                                      alg = 'JTVAE', 
                                      candidate_mode = 'random_one',    #random_one: randomly selece one candidate as target 
                                                                        #candidate_tree: build a candidadate tree

                                     ):
        n_nodes = len(uvisited)
        node_leaf = []
        dec_sml = 'CC'
        max_degree = 1
        maxd_node = None

        for node in moltree.nodes:
            node.is_leaf = (len(node.neighbors) == 1)

        try:
            while(sum(uvisited) > 1):   #leave one node as final
                node_leaf = []
                for i, node in enumerate(moltree.nodes):
                    if node.degree() == 1 and uvisited[node.idx] == 1:
                        node_leaf.append(node) 

                no_assemble = True

                for node in node_leaf:
                    if CNJMolAssembler.can_be_assembled(node):
                        CNJMolAssembler.assemble_one_node(node.neighbors[0], uvisited , alg, candidate_mode)

                        no_assemble = False
                        break

                if no_assemble:
                    #print('No valid node which can be assembled, using leaf-first algorithm!')
                    node  = node_leaf[0]
                    CNJMolAssembler.assemble_one_node(node, uvisited , alg, candidate_mode)
            
                for node in moltree.nodes:
                    node.is_leaf = (len(node.neighbors) == 1)

            idx = uvisited.index(1)
            final = moltree.nodes[idx]
        except Exception as e:
            print('assemble_JTMolTree_degree_first.Exception:')
            print(e.args)

        return final

    def assemble_order(moltree, use_stereo = False):
        #ref ExternalGraph\JTVAE\JTNN\JTNNVAE.py
        #decode(self, tree_vec, mol_vec, prob_decode)
        pred_root = moltree.nodes[0]
        pred_nodes = moltree.nodes

        for i, node in enumerate(pred_nodes):
            node.nid = i + 1  # ???why i+1?
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                ChemUtils.set_atommap(node.mol, node.nid)

        tree_mess = None  # JTNNEncoder
        mol_vec = None

        cur_mol = ChemUtils.copy_edit_mol(pred_root.mol)

        global_amap = [{}] + [{} for node in pred_nodes]  # [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
        global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in
                          cur_mol.GetAtoms()}  # [{}, {0: 0, 1: 1}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]	list

        cur_mol = CNJMolAssembler.dfs_assemble_without_score(tree_mess      = tree_mess, 
                                                             mol_vec        = mol_vec, 
                                                             all_nodes      = pred_nodes, 
                                                             cur_mol        = cur_mol,
                                                             global_amap    = global_amap, 
                                                             fa_amap        = [], 
                                                             cur_node       = pred_root, 
                                                             fa_node        = None,
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
                                   fa_node  #previous node, it mean which is father
                                   ):
        fa_nid = fa_node.nid if fa_node is not None else -1  # -1
        prev_nodes = [fa_node] if fa_node is not None else []  # []

        children = [nei for nei in cur_node.neighbors if  nei.nid != fa_nid]  # [0]:<ExternalGraph.JTVAE.JTNN.MolTree.MolTreeNode object at 0x0000023DA74AE048>
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]  # []
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]  # [0]:<ExternalGraph.JTVAE.JTNN.MolTree.MolTreeNode object at 0x0000023DA74AE048>
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid, a2, a1) for nid, a1, a2 in fa_amap if nid == cur_node.nid]  # []

        cands = ChemUtils.enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)  # 'C=O' + 'C' = 'O=[CH2:2]',
 
        n_random = 5
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
            #print('-------------------dfs_assemble return None------------------------')
            return None

        cand_smiles, cand_mols, cand_amap = zip(*cands)
        #print('dfs_assemble:cand_smiles', cand_smiles)

        cands = [(candmol, all_nodes, cur_node) for candmol in cand_mols]

        # -----------------------------------------------------------------------------

        if len(cands) == 1:
            cand_idx = [0]
        else:
            cand_idx = [*range(len(cands) - 1, 0, -1)]

        # -----------------------------------------------------------------------------------

        #RDKUtils.show_mol_with_atommap(cur_mol, atommap = False)

        backup_mol = Chem.RWMol(cur_mol)
        #for i in range(cand_idx.numel()):
        for i in range(len(cand_idx)):
            cur_mol = Chem.RWMol(backup_mol)
            #pred_amap = cand_amap[int(cand_idx[i].item())]  # [(2, 0, 0)]
            pred_amap = cand_amap[int(cand_idx[i])]  # [(2, 0, 0)]
            new_global_amap = copy.deepcopy(global_amap)  # [{}, {0: 0, 1: 1}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]

            for nei_id, ctr_atom, nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]  # [{}, {0: 0, 1: 1}, {0: 0}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]

            cur_mol = ChemUtils.attach_mols(cur_mol, children, [], new_global_amap)  # father is already attached
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            #RDKUtils.show_mol_with_atommap(new_mol, atommap = False)

            # print('dfs_assemble:new_mol',Chem.MolToSmiles(new_mol)) #jw  #'C=O'
            # Draw.MolsToGridImage([new_mol],    subImgSize=(600,600),).show()

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
                                                                     fa_node        = cur_node
                                                                     )
                if cur_mol_b is None:
                    result = False
                    break
                else:
                    cur_mol = cur_mol_b

            if result:
                return cur_mol

        #print('-------------------end of dfs_assemble------------------------')
        #return None
        return cur_mol

    def assemble_jtvae_decode(moltree):
        dec_smiles = ChemUtils.decode_moltree(moltree)
        return dec_smiles

    def assemble_JTMolTree(moltree,
                           alg = 'JTVAE', 
                           assemble_alg = 'JTVAE_decode',   #ChemUtils.decode_moltree
                                                            #group_first: randomly selece one candidate as target 
                                                            #leaf_first: build a candidadate tree
                           candidate_mode = 'random_one',   #random_one: randomly selece one candidate as target 
                                                            #candidate_tree: build a candidadate tree
                           final_select_alg = 'random',     #random: randomly select on candidate as final output
                                                            #plogp: maximum penalized_logp
                           use_stereo = True,               #wether enumerate  stereo  as part of final output candidate
                           ctoken = None,
                           p_mean = None,
                           ):
        #----------------------------------------------------------------
        #this method select node order by how many neighbors 
        #node with one neighbor is the first round, then remove this node from it's neighbor's  neighbor,
        #it means, it's neighbor remove one neighbor, then continue
        #so, the order of visited node is only base on how many neighbors, not the order of tree
        #different order generate different mols
        #so another method should be developed to assemble tree based on tree order
        #----------------------------------------------------------------
        n_nodes = len(moltree.nodes)
        uvisited = [1] * n_nodes #(range(0,n_nodes))
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

            if alg == 'JTVAE':
                for node in moltree.nodes:   #this could not used in BRICS, because if atommap is added then GetImplicitValence() get wrong value
                    if len(node.neighbors) > 1:
                        ChemUtils.set_atommap(node.mol, node.nid)

                if assemble_alg == 'JTVAE_decode':
                    final = CNJMolAssembler.assemble_order(moltree)
                    if final is None:
                        final = CNJMolAssembler.assemble_JTMolTree_degree_first(moltree, uvisited = uvisited,  alg = alg, candidate_mode = candidate_mode)   
                else:
                    raise 'Error!'
                #-------
                cur_mol =  CNJMolAssembler.get_final_mol(final, final_select_alg = final_select_alg,
                                                         use_stereo = use_stereo,
                                                         p_mean = p_mean,
                                                         )   
            
            else: #alg == 'BRICS':
                raise 'Error!'

            cur_mol = RDKUtils.remove_atommap_info_mol(cur_mol)
            dec_sml = Chem.MolToSmiles(cur_mol)
        except Exception as e:
            print(e.args)

        return dec_sml


    def get_final_mol(final_node, final_select_alg = 'random', use_stereo = True, p_mean = None):
        if isinstance(final_node, rdkit.Chem.rdchem.Mol):
            return final_node

        sml_cand = list(set([cand[0] for cand in final_node.candidates]))
        final_mol = None        

        if final_node.candidates == [] or final_select_alg == 'random':
            sml = final_node.smiles
            final_mol = Chem.MolFromSmiles(sml)
        else:
            raise 'Error!'

        return final_mol

    def Accemble_JTVAEMol(vocab,
                          ctoken,
                          bfs_binary_tree:CNJTMolTreeNode,
                          bfs_node_map,
                          show = True,
                          assemble_alg = 'JTVAE_decode',    #ChemUtils.decode_moltree
                                                            #group_first: randomly selece one candidate as target 
                                                            #leaf_first: build a candidadate tree
                          alg = 'JTVAE',
                          candidate_mode = 'random_one',    #random_one: randomly selece one candidate as target 
                                                            #candidate_tree: build a candidadate tree
                          final_select_alg = 'random',      #random: randomly select on candidate as final output
                                                            #plogp: maximum penalized_logp
                          use_stereo = True,                #wether enumerate  stereo  as part of final output candidate
                          p_mean = None
                          ):
        dec_smile = ''
        try:
            pred_root, pred_nodes, g, moltree = CNJMolAssembler.Convert_to_JTMoltree(vocab, ctoken,bfs_binary_tree, bfs_node_map, show)

            dec_smile = CNJMolAssembler.assemble_JTMolTree(moltree, 
                                                           alg = alg,
                                                           assemble_alg = assemble_alg,
                                                           candidate_mode   = candidate_mode,
                                                           final_select_alg = final_select_alg,
                                                           use_stereo       = use_stereo,
                                                           ctoken  = ctoken,
                                                           p_mean = p_mean,
                                                           )
        except Exception as e:
            print(e.args)
            
        return dec_smile




def test_reconstruct():
    print('-------------test_reconstruct-----------------')

    maxlen = 256
    vocab_file = r'H:\GitHub\t-SMILES\RawData\AID1706\active\Scaffold\active_smiles.smi.[Scaffold][24]_token.voc.smi'

    ctoken = CTokens(STDTokens_Frag_File(vocab_file), is_pad = True, pad_symbol = ' ', startend = True,
                     max_length = maxlen,  flip = False, invalid = True, onehot = False)    
    
    #-----------------------------------------------------------------------------------
    #sml = 'CC1=C(C=C(C=C1)OCC(=O)NC2=C(C=C(C=C2)C(=O)O)NC(=O)COC3=CC(=C(C=C3)C)C)C'

    bfs_ex = 'CC&O=C(COC1=CC=CC=C1)NC1=CC=CC=C1NC(=O)COC1=CC=CC=C1&CC&O=CO^CC&&&CC&CC&&'

    #----------------------------------------------------------------------------

    bfs_ex_smiles = CNJMolUtil.split_ex_smiles(bfs_ex, delimiter='^')
    print('bfs_ex_smiles', bfs_ex_smiles)
    
    bfs_tree, g, bfs_node_map =  CNJTMolTree.bfs_ex_reconstruct(ctoken = ctoken, bfs_ex_smiles = bfs_ex_smiles, show = True)

    if bfs_tree is not None:
        dec_smile =  CNJMolAssembler.Accemble_JTVAEMol(vocab            = None, #vocab,
                                                      ctoken            = ctoken,
                                                      bfs_binary_tree   = bfs_tree,
                                                      bfs_node_map      = None,
                                                      show              = True,
                                                      alg               = 'JTVAE',
                                                      #alg              = 'BRICS', #this should not be used now as linking to parts may wrong
                                                      assemble_alg      = 'JTVAE_decode',   #JTVAE_decode: ChemUtils.decode_moltree
                                                                                            #group_first: randomly selece one candidate as target 
                                                                                            #leaf_first: build a candidadate tree
                                                      candidate_mode    = 'random_one',     #random_one: randomly selece one candidate as target 
                                                                                            #candidate_tree: build a candidadate tree
                                                      final_select_alg  = 'plogp',          #random: randomly select on candidate as final output
                                                                                            #plogp: maximum penalized_logp
                                                      use_stereo = False,                   #wether enumerate  stereo  as part of final output candidate
                                                      )
        print(dec_smile)
        return 

def rebuild_file(n_samples = 1):
    maxlen = 256
    vocab_file = r'H:\GitHub\t-SMILES\RawData\AID1706\active\Scaffold\active_smiles.smi.[Scaffold][24]_token.voc.smi'

    ctoken = CTokens(STDTokens_Frag_File(vocab_file), is_pad = True, pad_symbol = ' ', startend = True,
                     max_length = maxlen,  flip = False, invalid = True, onehot = False)    
    
    #-----------------------------------------------------------------------------------

    smlfile = r'H:\GitHub\t-SMILES\RawData\AID1706\active\Scaffold\active_smiles.smi.[Scaffold][24]_join.csv'

    p_mean = None

    re_ex_smils = []
    re_smils = []
    errors = []
    new_vocs = []

  
    df = pd.read_csv(smlfile, squeeze=True, delimiter=',',header = None)       
    smiles_list = list(df.values)

    for i, bfs_ex_smiles in tqdm(enumerate(smiles_list), total = len(smiles_list),  desc = 'parsing smiles ...'):
        #bfs_ex_smiles shoule like ['CN', '&', 'NN', '&', 'C=N', '&', 'CC', '&', 'CC', '&', 'C[NH3+]', '&', 'C[NH3+]', '&', ...]
        #print('\r\n[bfs_ex_smiles]: ', bfs_ex_smiles)

        if not isinstance(bfs_ex_smiles, (str)):
            print(f'[test_rebuild_file is not a string-{i}:-{bfs_ex_smiles}]')
            continue
        else:
            bfs_ex_smiles = CNJMolUtil.split_ex_smiles(bfs_ex_smiles, delimiter='^')

        if bfs_ex_smiles is None or len(bfs_ex_smiles) < 1:
            print(f'[test_rebuild_file is null or empty-{i}:-{bfs_ex_smiles}]')
            continue

        if len(bfs_ex_smiles) > ctoken.max_length:
            bfs_ex_smiles = bfs_ex_smiles[0:ctoken.max_length]


        bfs_tree, g, bfs_node_map =  CNJTMolTree.bfs_ex_reconstruct(ctoken = ctoken, bfs_ex_smiles = bfs_ex_smiles, show = False)

        if bfs_tree is not None:                      
            bfs_ex_nodeid, bfs_ex_vocids, bfs_ex_smiles, new_voc= CNJTMolTree.get_bfs_ex(ctoken, bfs_tree)  #generate advanced bfs 
            if len(bfs_ex_smiles) % 2 == 1:
                bfs_ex_smiles.append['&']

            if bfs_ex_smiles is not None:
                re_ex_smils.append(bfs_ex_smiles)

            if len(new_voc)> 0:
                new_vocs.append(new_voc)

            for k in range(n_samples):
                dec_smile =  CNJMolAssembler.Accemble_JTVAEMol(vocab = None,    #vocab,
                                                              ctoken = ctoken,
                                                              bfs_binary_tree = bfs_tree,
                                                              bfs_node_map = bfs_node_map,
                                                              show = False,
                                                              alg = 'JTVAE',
                                                              #alg = 'Brics',  #this should not be used now as linking to parts may wrong
                                                              assemble_alg      = 'JTVAE_decode',    #JTVAE_decode:
                                                                                                    #group_first: randomly selece one candidate as target 
                                                                                                    #leaf_first: build a candidadate tree
                                                              candidate_mode    = 'random_one',     #random_one: randomly selece one candidate as target 
                                                                                                    #candidate_tree: build a candidadate tree
                                                              final_select_alg  = 'plogp',          #random: randomly select on candidate as final output
                                                                                                    #plogp: maximum penalized_logp
                                                              use_stereo = False,                   #wether enumerate  stereo  as part of final output candidate
                                                              p_mean = p_mean,
                                                              )
                if dec_smile is not None and len(dec_smile)>0:
                    re_smils.append(dec_smile)
                else:
                    errors.append(bfs_ex_smiles)
       
    #output = smlfile + f'.re_ex_smils.csv'
    #df = pd.DataFrame(re_ex_smils)
    #df.to_csv(output, index = False, header=False, na_rep="NULL")
  
    output = smlfile + f'.re_smils.smi'
    df = pd.DataFrame(re_smils)
    df.to_csv(output, index = False, header=False, na_rep="NULL")

    output = smlfile + f'.errors.csv'
    df = pd.DataFrame(errors)
    df.to_csv(output, index = False, header=False, na_rep="NULL")
    
    new_vocs = [x for l in new_vocs for x in l]  #convert list(list(string)) to list(string)
    new_vocs.sort()
    output = smlfile + f'.new_vocs.smi'
    df = pd.DataFrame(list(set(new_vocs)))
    df.to_csv(output, index = False, header=False, na_rep="NULL")
      
    print(output)

    return

if __name__ == '__main__':

    test_reconstruct()

    #rebuild_file()
