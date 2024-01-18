import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx

import rdkit.Chem as Chem

from Tools.GraphTools import GTools
from Tools.MathUtils import BCMathUtils
from DataSet.STDTokens import CTokens
from DataSet.JTNN.MolTree import Vocab, MolTree
from DataSet.STDTokens import CTokens, STDTokens_Frag_File
from DataSet.Graph.CNJMolUtil import CNJMolUtil
from DataSet.Graph.CNXBinaryTree import CNXBinaryTree        
from MolUtils.RDKUtils.Frag.RDKFragUtil import Fragment_Alg

class CNJTMolTreeNode():
    disconnect_char = '^'  
    invalid_char = '&'

    def __init__(self, idx, nx_node):
        super(CNJTMolTreeNode, self).__init__()
        self.idx = idx
        self.data = nx_node

        self.left = None
        self.right = None
        self.parent = None
        self.level = 0

        return

    def add_left(self, node):
        if self.parent is not None:
            self.level = self.parent.level + 1
        else:
            self.level = 0

        node.parent = self
        self.left = node
        
        return self.left

    def add_right(self, node):
        if self.parent is not None:
            self.level = self.parent.level + 1
        else:
            self.level = 0

        node.parent = self
        self.right = node

        return self.right

    def add_parent(self, parent, is_left):
        self.parent = parent
        if is_left:
            parent.add_left(self)
        else:
            parent.add_right(self)

        return

     

class CNJTMolTree(MolTree):        
    def __init__(self, smiles,
                 jtvoc = None,
                 ctoken = None, 
                 dec_alg = Fragment_Alg.MMPA_DY,
                 kekuleSmiles = True,
                 ) -> None:

        self.combine_ex_smiles = None
        self.skeleton = None
        self.dec_alg = dec_alg
        self.kekuleSmiles   = kekuleSmiles 

        self.ctoken         = ctoken
        self.end_token      = ctoken.invalid_token #'&'
        self.end_token_idx  = ctoken.invalid_index # #P

        self.jtvoc = ctoken.STDTokens.vocab
        self.n_voc = self.jtvoc.size()      

        if smiles is not None and len(smiles) > 0:
            self.init_from_smile(smiles, jtvoc,ctoken, dec_alg = dec_alg, kekuleSmiles = kekuleSmiles)
        else:
            #print('create a dummy CNJTMolTree')
            self.mol = None

        return

    def init_from_smile(self, 
                        smiles:str,  
                        jtvoc: Vocab,
                        ctoken: CTokens,
                        dec_alg = 'BRICS',
                        kekuleSmiles = True,
                        ):
        try:
            super(CNJTMolTree, self).__init__(smiles = smiles, dec_alg = dec_alg, kekuleSmiles = kekuleSmiles)
               
            if self.mol is None:
                return 

            self.n_nodes = len(self.nodes)

            self.neighbor_map = []
            self.nx_parent_map = {}
            self.nx_child_map = {}
            self.nx_parent_map[0] = 0

            self.graph_nx = self.convert_to_nx(show = False)

            self.bfs_idx = GTools.BFS(self.graph_nx)      
            self.dfs_idx = GTools.DFS(self.graph_nx )  


            self.build_relation(self.bfs_idx)

            self.encoder_joint_points()

            self.bfs_binary_tree = self.create_bfs_binary_tree_ex(show = False)

            self.bfs_ex_nodeid, self.bfs_ex_vocids, self.bfs_ex_smiles, self.new_vocs, self.bfs_ex_smarts = CNJTMolTree.get_bfs_ex(self.ctoken, 
                                                                                                                                   self.bfs_binary_tree, 
                                                                                                                                   extra_dummy = True)   #generate advanced bfs 

            self.bfs_idx_ex = GTools.BFS(self.nx_binarytree)       
            self.dfs_idx_ex = GTools.DFS(self.nx_binarytree )   

            if self.bfs_ex_vocids is None:
                self.mol = None
            self.combine_ex_smiles, self.skeleton = CNJMolUtil.combine_ex_smiles(self.bfs_ex_smiles)
            self.combine_ex_smarts, _             = CNJMolUtil.combine_ex_smiles(self.bfs_ex_smarts)

        except Exception as e:
            print('[CNJTMolTree.init_from_smile].Exception:', e.args)
            print('[CNJTMolTree.init_from_smile].Exception:', smiles)
            self.mol = None

        return 

    def encoder_joint_points(self):
        bfs_idx = self.bfs_idx
        bfs_nodes = []     
        dmy_id = 0
        for i, cbd in enumerate(self.cut_bonds):
            at_l = f'[{cbd[0]}*]'
            at_r = f'[{cbd[1]}*]'

            edge = self.edges[i]
            node_l = self.nodes[edge[0]]
            node_r = self.nodes[edge[1]]

            if node_l.smarts.find(at_l) != -1:
                node_l.smarts = node_l.smarts.replace(at_l, f'[x{dmy_id}*]')
                node_r.smarts = node_r.smarts.replace(at_r, f'[x{dmy_id}*]')
                dmy_id+= 1
            elif node_l.smarts.find(at_r) != -1:
                node_l.smarts = node_l.smarts.replace(at_r, f'[x{dmy_id}*]')
                node_r.smarts = node_r.smarts.replace(at_l, f'[x{dmy_id}*]')
                dmy_id+= 1
            else:
                print('[encoder_joint_points].Error:', cbd, self.smiles )

        for node in self.nodes:
            node.smarts = node.smarts.replace('[x','[')
                    
        for nidx in self.graph_nx.nodes:
            nx_node = self.graph_nx.nodes[nidx]
            nx_node['smarts'] = nx_node['jtnode'].smarts 

        return 

    def build_relation(self, bfs_idx):               
        visited = np.zeros((self.n_nodes))
        visited[0] = 1
        for idx in bfs_idx:
            pid = idx
            nbs = (list(self.neighbor_map[idx])).copy()
            for nb in nbs:
                if visited[nb] == 1:
                    pid = nb
                    break                

            visited[idx] = 1
            self.nx_parent_map[idx] = pid   #

        for idx in bfs_idx:
            nbs = (list(self.neighbor_map[idx])).copy()
            pidx = self.nx_parent_map[idx]

            index = BCMathUtils.find_index(nbs, pidx)
            if index != -1:
                nbs.pop(index)

            self.nx_child_map[idx] = nbs
        return 


    def find_parent_node(self, idx, visited):
        nbs = self.neighbor_map[idx]

        pid = -1
        parent = None
        for i in nbs:
            if visited[i] == 1:
                pid = i
                break
        if pid != -1:
            parent = self.bfs_node_map[pid]
        else:
            parent = self.bfs_node_map[idx] 

        self.nx_parent_map[idx] = parent.idx   
        return parent

    def create_bfs_binary_tree_ex(self, show = False):
        self.nx_binarytree = CNXBinaryTree()  #DiGraph
         
        visited = np.zeros((self.n_nodes))

        self.bfs_idx_ex = []
        self.bfs_node_map = {}   
    
        idx = 0
        nx_node = self.graph_nx.nodes[idx]
        self.bfs_binary_tree = CNJTMolTreeNode(idx=idx, nx_node=nx_node)  
        self.bfs_node_map[idx] = self.bfs_binary_tree
        visited[idx] = 1

        point = self.bfs_binary_tree
        point.level = 0
        point.parant = point

        for idx in self.bfs_idx:
            nx_node = self.graph_nx.nodes[idx]
            nb_nodes = self.graph_nx.neighbors(idx)

            bfs_node = CNJTMolTreeNode(idx=idx, nx_node=nx_node)
            self.bfs_node_map[idx] = bfs_node
            self.nx_binarytree.add_node(idx, 
                                        nid   = idx,
                                        smile = nx_node['smile'],
                                        data  = bfs_node,
                                        smarts = nx_node['smarts'],
                                        )              

        for idx in self.nx_child_map:
            nbs = self.nx_child_map[idx]
            if len(nbs) == 0: 
                 visited[idx]  = 1
            else:
                child_idx = nbs[0]
                p_idx = self.nx_parent_map[child_idx]

                parent = self.nx_binarytree.nodes[p_idx]['data']
                child = self.nx_binarytree.nodes[child_idx]['data']

                self.nx_binarytree.add_edge_left(p_idx, child_idx)  #nx_binarytree left

                point = parent.add_left(child)  

                visited[child_idx]  = 1

                for i in range(1, len(nbs)):
                    b_idx = nbs[i]
                    self.nx_binarytree.add_edge_right(point.idx, b_idx)  

                    brother = self.nx_binarytree.nodes[b_idx]['data']
                    point = point.add_right(brother) 

                    visited[b_idx]  = 1

        self.nx_binarytree.make_full()

        for idx in self.bfs_idx:
            self.bfs_node_map[idx] = self.nx_binarytree.nodes[idx]['data']


        self.bfs_binary_tree = self.nx_binarytree.nodes[0]['data']
        return self.bfs_binary_tree 


    def create_bfs_binary_tree(self, show=False):  #
        #input  : self.bfs_idx which comes from self.graph_nx which comes from CNJTMolTree
        #output : self.nx_binarytree which is a nx.DiGraph
        #       : self.bfs_node_map which map idx to CNJTMolTreeNode

        self.bfs_idx_ex = []
        self.nx_binarytree = CNXBinaryTree()

        visited = np.zeros((self.n_nodes))
        self.bfs_node_map = {}   

        idx = 0
        nx_node = self.graph_nx.nodes[idx]
        self.bfs_binary_tree = CNJTMolTreeNode(idx=idx, nx_node=nx_node)  #the first node
        self.bfs_node_map[idx] = self.bfs_binary_tree
        visited[idx] = 1

        point = self.bfs_binary_tree
        point.level = 0

        if len(self.bfs_idx) == 1:
            self.nx_binarytree.add_node(0, 
                                       smile  = self.bfs_node_map[0].data['smile'],
                                       nid  = 0,
                                       smarts = self.bfs_node_map[0].data['smarts'],
                                       )

        for i in range(0, len(self.bfs_idx)) :
            idx = self.bfs_idx[i]
            neighbors = self.neighbor_map[idx]

            if visited[idx] == 1 and len(neighbors) <= 2: 
                continue

            parent = self.find_parent_node(idx, visited)
            if parent.idx != idx:
                nx_node = self.graph_nx.nodes[idx]
                bfs_node = CNJTMolTreeNode(idx=idx, nx_node=nx_node)
                point = parent.add_left(bfs_node)                            
                self.bfs_node_map[idx] = bfs_node
                visited[idx] = 1

                self.nx_binarytree.add_edge_left(parent.idx, bfs_node.idx)  

            first = True
            for nb in neighbors:
                if visited[nb] == 1:
                    continue

                visited[nb] = 1
                nx_node = self.graph_nx.nodes[nb]
                bfs_node = CNJTMolTreeNode(idx=nb, nx_node=nx_node)
                self.bfs_node_map[nb] = bfs_node

                if first:
                    nbpoint = point.add_left(bfs_node)   
                    first = False
                    self.nx_binarytree.add_edge_left(point.idx, bfs_node.idx) 
                else:
                    self.nx_binarytree.add_edge_right(nbpoint.idx, bfs_node.idx)                      
                    nbpoint = nbpoint.add_right(bfs_node)  
                                    

        for node in self.nx_binarytree.nodes:
            idx = node
            if isinstance(idx, CNJTMolTreeNode):
                continue
            else:
                self.nx_binarytree.nodes[idx]['nid'] = idx
                self.nx_binarytree.nodes[idx]['smile'] = self.bfs_node_map[idx].data['smile']

        self.nx_binarytree.make_full()

                    
        return self.bfs_binary_tree

    def get_bfs_ex(ctoken, bfs_binary_tree, extra_dummy = True):
        queue = []

        point = bfs_binary_tree
        queue.append(point)
       
        bfs_ex_nodeid = []
        bfs_ex_vocid = []
        bfs_ex_smiles = []
        bfs_ex_smarts = []
        new_vocs = []

        try:
            while len(queue) != 0:
                item = queue[0]
                if item is not None:
                    queue.append(item.left)
                    queue.append(item.right)

                if item is not None:
                    if item.data['idx_voc'] == -1:
                        sml = item.data['smile']
                        if Chem.MolFromSmiles(sml) is not None:
                            new_vocs.append(sml)

                    bfs_ex_nodeid.append(item.idx)  
                    bfs_ex_vocid.append(item.data['idx_voc'])
                    bfs_ex_smiles.append(item.data['smile'])
                    bfs_ex_smarts.append(item.data['smarts'])
                else:
                    bfs_ex_nodeid.append('p')

                    bfs_ex_vocid.append(ctoken.invalid_index)
                    bfs_ex_smiles.append(ctoken.invalid_token)
                    bfs_ex_smarts.append(ctoken.invalid_token)

                queue.pop(0)
            #end while
            
            if extra_dummy:
                bfs_ex_nodeid.insert(1, 'p')
                bfs_ex_vocid.insert(1, ctoken.invalid_index)
                bfs_ex_smiles.insert(1, ctoken.invalid_token)
                bfs_ex_smarts.insert(1, ctoken.invalid_token)

            return bfs_ex_nodeid, bfs_ex_vocid, bfs_ex_smiles, new_vocs, bfs_ex_smarts
        except Example as e:
            print('[CNJTMolTree.get_bfs_ex].Exception', e.args)
            return None, None, None, None, None

    def bfs_ex_reconstruct(ctoken, bfs_ex_id = None, bfs_ex_smiles = None, clean_up = True, show = False):
        if bfs_ex_id is None and bfs_ex_smiles is None:
            raise ValueError('[CNJTMol-bfs_ex_reconstruct]: input could not be None!')
            return None, None        
        
        nlen = len(bfs_ex_smiles)
        if nlen == 0 :
            return None, None, None

        for i in range(len(bfs_ex_smiles)):
            if not CNJMolUtil.is_dummy(bfs_ex_smiles[i]):                
                bfs_ex_smiles[i] = CNJMolUtil.valid_smiles(bfs_ex_smiles[i], ctoken = ctoken)

        if nlen >  1 and bfs_ex_smiles[1] == ctoken.invalid_token: 
            bfs_ex_smiles.pop(1)

        if bfs_ex_id is None:
            bfs_ex_id = []
            for sml in bfs_ex_smiles:
                idx = ctoken.get_index(sml)
                bfs_ex_id.append(ctoken.get_index(sml))
   
        if bfs_ex_smiles is None:
            bfs_ex_smiles = []
            for token_id in bfs_ex_id:
                bfs_ex_smiles.append(ctoken.get_token(token_id))

                        
        edged_ids = []
        dummy_ids = []
        dummy_nodes = []

        g = CNXBinaryTree()
        for i, token_id in enumerate(bfs_ex_id):
            g.add_node(i, 
                       smile  = bfs_ex_smiles[i], 
                       clique = None, 
                       nid  = i,  
                       is_leaf = False,
                       idx_voc = bfs_ex_id[i],
                       jtnode = None,
                       smarts = bfs_ex_smiles[i],
                       )
                       
            if bfs_ex_smiles[i]  == '&':
                dummy_nodes.append(i)

        bfs_node_map = {}

        queue = []
        idx = 0
        t_id = bfs_ex_id[0]
        sml = bfs_ex_smiles[0]
        bfs_tree = CNJTMolTreeNode(idx = idx, nx_node = g.nodes[idx])  
        bfs_node_map[idx] = bfs_tree
        idx+=1

        edged_ids.append(0)

        if nlen == 1 or nlen == 2:
            return bfs_tree, g, bfs_node_map
   
        queue.append(bfs_tree)

        t_id_1 = bfs_ex_id[1]
        sml_1 = bfs_ex_smiles[1]
        if sml_1 == ctoken.invalid_token:
            idx += 1

        try:
            for i in range(idx, nlen, 2):
                if len(queue) ==0:
                    break
                parent = queue.pop(0)
                if parent.data['idx_voc'] == ctoken.invalid_index:
                    break

                tid_left = bfs_ex_id[i]
                sml_left = bfs_ex_smiles[i]
   
                tid_right = bfs_ex_id[i+1]
                sml_right = bfs_ex_smiles[i+1]

                node_left = CNJTMolTreeNode(idx = idx, nx_node=g.nodes[idx]) 
                bfs_node_map[idx] = node_left
                idx+=1
                node = parent.add_left(node_left)
                g.add_edge_left(parent.idx, node.idx) 
                edged_ids.append(parent.idx)
                edged_ids.append(node.idx)

                node_right = CNJTMolTreeNode(idx = idx, nx_node=g.nodes[idx])                     
                bfs_node_map[idx] = node_right
                idx+=1
                node = parent.add_right(node_right)
                g.add_edge_right(parent.idx, node.idx) 
                edged_ids.append(parent.idx)
                edged_ids.append(node.idx)

                if tid_left != ctoken.invalid_index:
                    queue.append(node_left)
                if tid_right != ctoken.invalid_index:
                    queue.append(node_right)                

        except Exception as e:
            msg = e.args
  
        if clean_up:
            edged_ids = set(edged_ids)
            all_ids = set([*range(0, len(g.nodes), 1)])
            dummy_ids = list(all_ids - edged_ids)

            dummy_ids.sort(reverse = True)
            for idx in dummy_ids:
                    g.remove_node(idx)

        i = 0
        id_map = {}
        for idx in bfs_node_map:
            id_map[idx] = i
            i+=1
            
        id_map_rvs = {key: val for key, val in sorted(id_map.items(), key = lambda ele: ele[0], reverse = False)}

        for key in id_map_rvs:
            item = bfs_node_map.pop(key)
            item.idx = id_map_rvs[key]
            bfs_node_map[id_map_rvs[key]] = item

        g = nx.relabel_nodes(g, id_map_rvs)
        for nid in g.nodes():
            g.nodes[nid]['nid'] = nid

          
        return bfs_tree, g, bfs_node_map


    def convert_to_nx(self, show = False):
        g = nx.Graph()
        n_nodes = len(self.nodes)

        for i, node in enumerate(self.nodes):
            g.add_node(i, 
                       smile = node.smiles, 
                       clique = node.clique, 
                       nid = node.nid,  
                       is_leaf = node.is_leaf,
                       idx_voc = self.ctoken.get_index(node.smiles),
                       jtnode = node,
                       smarts = node.smarts,
                       )

        for i, node in enumerate(self.nodes):
            start = node.nid - 1
            nblist = node
            nbs = []
            for nb in node.neighbors:
                end = nb.nid - 1
                g.add_edge(start, end)
                nbs.append(end)

        span_g = nx.minimum_spanning_tree(g)  
        if len(g.edges) != len(span_g.edges):
            print('[convert_to_nx] remove circle:', self.smiles)

        self.edges = list(span_g.edges)
        self.neighbor_map = [()] * len(self.nodes)

        for (s,e) in self.edges:
            sbm = set(self.neighbor_map[s])
            sbm.add(e)
            self.neighbor_map[s] = sbm

            sbm = set(self.neighbor_map[e])
            sbm.add(s)
            self.neighbor_map[e] = sbm

        for i, node in enumerate(self.nodes):
            nbm = self.neighbor_map[i]      
            node.neighbors = []
            for nid in nbm:
                node.neighbors.append(self.nodes[nid])

            node.is_leaf = (len(node.neighbors) == 1)        

        return span_g      


class CNJMolUtils:
    def encode_single(smls, ctoken, dec_alg):
        #print('[smls is]:', smls)
        try:
            sub_smils = smls.strip().split('.')  #for reaction
            combine_sml = []
            combine_smt = []
            for sml in sub_smils:
                cnjtmol = CNJTMolTree(sml, ctoken = ctoken, dec_alg = dec_alg)
                if cnjtmol is  None:
                    print('cnjtmol is None')
     
                bfs_ex_smiles = cnjtmol.bfs_ex_smiles
                bfs_ex_smarts = cnjtmol.bfs_ex_smarts

                atom_env = cnjtmol.atom_env
                s, skeleton = CNJMolUtil.combine_ex_smiles(bfs_ex_smiles)
                combine_sml.append(s)
     
                s, skeleton = CNJMolUtil.combine_ex_smiles(bfs_ex_smarts)
                combine_smt.append(s)

                words = CNJMolUtil.split_ex_smiles(s)

            combine_sml = '.'.join(s for s in combine_sml)  
            combine_smt = '.'.join(s for s in combine_smt)
        except Exception as e:
            print('[CNJMolUtils.encode_single].exception:', e.args)
            combine_sml = 'CC&&&'
            combine_smt = 'CC&&&'

        return combine_sml, combine_smt


def preprocess():
    

    dec_algs = [
                Fragment_Alg.Vanilla,
                Fragment_Alg.JTVAE,
                Fragment_Alg.BRICS,
                Fragment_Alg.BRICS_Base,
                Fragment_Alg.MMPA,
                Fragment_Alg.Scaffold,
                Fragment_Alg.BRICS_DY,
                Fragment_Alg.MMPA_DY,
                Fragment_Alg.Scaffold_DY,
                #Fragment_Alg.RBrics_DY,
                ]

    #kekuleSmiles = False
    kekuleSmiles = True
  
    ctoken = CTokens(STDTokens_Frag_File(None), max_length = 256,   invalid = True, onehot = False)

    smlfile = r'../RawData/Chembl/Test/Chembl_test.smi'
    print(smlfile)
    df = pd.read_csv(smlfile, squeeze=True, delimiter=',',header = None, skip_blank_lines = True) 
    smiles_list = list(df.values)

    for dec_alg in dec_algs:
        bfs_ex_list = []
        bfs_smile_list = []
        bfs_smile_list_join = []
        bfs_smarts_list_join = []
        org_smile_list = []
        bfs_max_Len = 0
        vocab_list = set()
        skt_list = set()
        atom_env_list = []

        save_prex = dec_alg.name

        for i, sml in tqdm(enumerate(smiles_list), total = len(smiles_list),  desc = 'parsing smiles ...'):
            #print(sml)
            if sml is None or str(sml) == 'nan' :
                continue
            sml = ''.join(sml.strip().split(' '))

        
            sub_smils = sml.strip().split('.')  
            try:
                org_smile_sub = ''
                bfs_smile_list_sub = []
                bfs_smart_list_sub = []
                joined_sub_smiles = ''
                joined_sub_smarts = ''
                skeleton_sub = ''

                joined_smiles = None
                for i, sub_s in enumerate(sub_smils):
                    cnjtmol = CNJTMolTree(sub_s, ctoken = ctoken, dec_alg = dec_alg) 

                    if cnjtmol.mol is not None:
                        for c in cnjtmol.nodes:
                            vocab_list.add(c.smiles)


                        joined_smiles, skeleton = CNJMolUtil.combine_ex_smiles(cnjtmol.bfs_ex_smiles)
                        joined_smarts, _        = CNJMolUtil.combine_ex_smiles(cnjtmol.bfs_ex_smarts)
  
                        if i > 0:
                            org_smile_sub   += '.'
                            bfs_smile_list_sub.extend(['.'])
                            bfs_smart_list_sub.extend(['.'])
                            joined_sub_smiles   += '.'
                            joined_sub_smarts   += '.'
                            skeleton_sub        += '.'

                        org_smile_sub = org_smile_sub + sub_s
                        bfs_smile_list_sub.extend(cnjtmol.bfs_ex_smiles)
                        bfs_smart_list_sub.extend(cnjtmol.bfs_ex_smarts)

                        joined_sub_smiles = joined_sub_smiles + joined_smiles
                        joined_sub_smarts = joined_sub_smarts + joined_smarts

                        skeleton_sub = skeleton_sub + skeleton
                        
                        if cnjtmol.atom_env is not None:
                            atom_env_list.extend(cnjtmol.atom_env)

                if joined_smiles is None:
                    continue

                joined_smiles = joined_smiles.strip()

                org_smile_list.append(org_smile_sub)
                bfs_smile_list.append(bfs_smile_list_sub)     
                
                bfs_smile_list_join.append(joined_sub_smiles)
                bfs_smarts_list_join.append(joined_sub_smarts)

                skt_list.add(skeleton_sub)
                
                if len(joined_sub_smarts) > bfs_max_Len:
                    bfs_max_Len = len(joined_sub_smarts)

            except Exception as e:
                print('[CNJTMol.preprocess].Exception:', e.args)
                print('[CNJTMol.preprocess].Exception:', sml)
                continue
    
       
        ctoken.maxlen = bfs_max_Len + 4  #
       

        vocab = sorted(vocab_list)
        output = smlfile + f'.[{save_prex}][{bfs_max_Len}]_token.voc.smi'
        df = pd.DataFrame(vocab)
        df.to_csv(output, index = False, header=False, na_rep="NULL")

        output = smlfile + f'.[{save_prex}][{bfs_max_Len}]_org.csv'
        df = pd.DataFrame(org_smile_list)
        df.to_csv(output, index = False, header=False, na_rep="NULL")

        skt_list = list(skt_list)
        skt_list.sort()
        output = smlfile + f'.[{save_prex}][{bfs_max_Len}]_skt.csv'
        df = pd.DataFrame(skt_list)
        df.to_csv(output, index = False, header=False, na_rep="NULL")
        print('[skt_list]:',len(skt_list))

        if dec_alg in [Fragment_Alg.JTVAE,
                       Fragment_Alg.BRICS,
                       Fragment_Alg.BRICS_Base,
                       Fragment_Alg.MMPA,
                       Fragment_Alg.Scaffold
                       ]:
            output = smlfile + f'.[{save_prex}][{bfs_max_Len}]_TSSA.csv'
            df = pd.DataFrame(bfs_smile_list_join)
            df.to_csv(output, index = False, header=False, na_rep="NULL")
        elif dec_alg in [Fragment_Alg.BRICS_DY,                          
                         Fragment_Alg.MMPA_DY, 
                         Fragment_Alg.Scaffold_DY , 
                         Fragment_Alg.RBrics_DY
                         ]:
            output = smlfile + f'.[{save_prex}][{bfs_max_Len}]_TSDY.csv'
            df = pd.DataFrame(bfs_smile_list_join)
            df.to_csv(output, index = False, header=False, na_rep="NULL")
   
            output = smlfile + f'.[{save_prex}][{bfs_max_Len}]_TSID.csv'
            df = pd.DataFrame(bfs_smarts_list_join)
            df.to_csv(output, index = False, header=False, na_rep="NULL")
        else: #Fragment_Alg.Vanilla
            output = smlfile + f'.[{save_prex}][{bfs_max_Len}]_TSV.csv'
            df = pd.DataFrame(bfs_smile_list_join)
            df.to_csv(output, index = False, header=False, na_rep="NULL")


    return
