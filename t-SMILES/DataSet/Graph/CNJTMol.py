import re
import random

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
        self.end_token      = ctoken.invalid_token 
        self.end_token_idx  = ctoken.invalid_index 

        self.jtvoc = ctoken.STDTokens.vocab
        self.n_voc = self.jtvoc.size()      

        if smiles is not None and len(smiles) > 0:
            self.init_from_smile(smiles, jtvoc,ctoken, dec_alg = dec_alg, kekuleSmiles = kekuleSmiles)
        else:
            self.mol = None

        return

    def init_from_smile(self, 
                        smiles:str,  
                        jtvoc: Vocab,
                        ctoken: CTokens,
                        dec_alg = 'BRICS',
                        kekuleSmiles = True,
                        skt_random = False,  #reorder FBT
                        frg_random = False,  #reorder smiles of frag 
                        standardize = False,
                        ):
        try:
            if standardize:
                smiles = RKDMol.standardize(smiles)

            super(CNJTMolTree, self).__init__(smiles = smiles, dec_alg = dec_alg, kekuleSmiles = kekuleSmiles, frg_random = frg_random)
               
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


            self.amt_bfs_smiles, self.amt_bfs_smarts  = self.get_amt_bfs_smiles()
            self.amt_dfs_smiles, self.amt_dfs_smarts  = self.get_amt_dfs_smiles()

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
    def get_amt_bfs_smiles(self):
        amt_bfs_smiles = []
        amt_bfs_smarts = []

        try:
            if self.bfs_idx is not None and len(self.bfs_idx) > 0:
                for idx in self.bfs_idx:
                    node = self.graph_nx.nodes[idx]
                    sml = node['smile']
                    smt = node['smarts']
                    amt_bfs_smiles.append(sml)
                    amt_bfs_smarts.append(smt)
        except Exception as e:
            print('[CNJTMolTree.get_amt_bfs_smiles].Exception:', e.args)

        amt_bfs_smiles = '^'.join(amt_bfs_smiles)
        amt_bfs_smarts = '^'.join(amt_bfs_smarts)
        return amt_bfs_smiles, amt_bfs_smarts

    def get_amt_dfs_smiles(self):
        amt_smiles = []
        amt_smarts = []

        try:
            if self.dfs_idx is not None and len(self.dfs_idx) > 0:
                for idx in self.dfs_idx:
                    node = self.graph_nx.nodes[idx]
                    sml = node['smile']
                    smt = node['smarts']
                    amt_smiles.append(sml)
                    amt_smarts.append(smt)
        except Exception as e:
            print('[CNJTMolTree.get_amt_bfs_smiles].Exception:', e.args)

        amt_smiles = '^'.join(amt_smiles)
        amt_smarts = '^'.join(amt_smarts)
        return amt_smiles, amt_smarts

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
        self.nx_parent_map = {}
        self.nx_child_map = {}
        self.nx_parent_map[0] = 0

        visited = np.zeros((self.n_nodes))
        
        idx = bfs_idx[0]
        visited[idx] = 1
        for idx in bfs_idx:
            pid = idx
            nbs = (list(self.neighbor_map[idx])).copy()
            for nb in nbs:
                if visited[nb] == 1:
                    pid = nb
                    break                

            visited[idx] = 1
            self.nx_parent_map[idx] = pid   

        for idx in bfs_idx:
            nbs = (list(self.neighbor_map[idx])).copy()
            pidx = self.nx_parent_map[idx]

            index = BCMathUtils.find_index(nbs, pidx)
            if index != -1:
                nbs.pop(index)

            self.nx_child_map[idx] = nbs
        return 

    def find_visited_brother(self, node, parent, visited):
        idx = parent.idx
        nb_nodes = self.graph_nx.neighbors(idx)

        point = self.bfs_node_map[idx]  

        nbs = (list(self.neighbor_map[idx])).copy()

        if point.parent is None:
            return None
        else:
            if idx in self.nx_parent_map:
                index = self.nx_parent_map[idx]
            else:
                index = -1
            #index = self.nx_parent_map
            #index = BCMathUtils.find_index(nbs, point.parent.idx)
            if index != -1:
                nbs.pop(index)

        pid = -1
        last_visited = -1
        brother = None
        i = 0

        index = BCMathUtils.find_index(self.bfs_idx, idx)
        i = index
        while i > 0:
            i -= 1
            pre = self.bfs_idx[i]
            if pre in nbs:
                last_visited = pre

            #if visited[nbs[i]] == 1:
            #    pid = nbs[i]
            #    break         


        if last_visited != -1:
             brother = self.bfs_node_map[last_visited]

        #if pid != -1 and pid in self.bfs_node_map:
        #    brother = self.bfs_node_map[pid]
        #else:
        #    brother = self.bfs_node_map[idx]  #parent is itself

        return brother

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
        self.nx_binarytree = CNXBinaryTree()  
         
        visited = np.zeros((self.n_nodes))

        self.bfs_idx_ex = []
        self.bfs_node_map = {}   
    
        #idx = 0
        idx = self.bfs_idx[0]
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

                self.nx_binarytree.add_edge_left(p_idx, child_idx)  

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


        start_pos = self.bfs_idx[0]
        self.bfs_binary_tree = self.nx_binarytree.nodes[start_pos]['data']

        return self.bfs_binary_tree #the first node


    def binarytree_to_tree(bfs_binary_tree, clean_up = True, show = False): #works fine
        #self.nx_binarytree
        #bfs_binary_tree = self.nx_binarytree.nodes[0]['data']
        g = nx.DiGraph()

        queue = []
        point = bfs_binary_tree
        queue.append(point)
       
        bfs_ex_nodeid = []
        node_map = {}
        dummy_ids = []

        while len(queue) != 0:
            item = queue.pop(0)

            if item is not None:
                queue.append(item.left)
                queue.append(item.right)

            if item is not None:  
                bfs_ex_nodeid.append(item.idx)  #to verify 
                node_map[item.idx] = item
                
                g.add_node(item.idx, 
                            nid   = item.idx,
                            smile = item.data['smile'],
                            data  = item,
                            smarts = item.data['smarts'],
                            )
                if item.data['smile']  == '&':
                    dummy_ids.append(item.idx)

        for idx in bfs_ex_nodeid:
            node  = node_map[idx]
            left = node.left
            if left is not None:
                if clean_up:
                    if left.data['smile'] != '&':
                        g.add_edge(node.idx, left.idx)
                else:
                    g.add_edge(node.idx, left.idx)

                right = left.right                
                while right is not None:
                    if clean_up:
                        if right.data['smile'] != '&':
                            g.add_edge(node.idx, right.idx)
                    else:
                        g.add_edge(node.idx, right.idx)

                    right = right.right

        #clean up
        if clean_up:
            dummy_ids.sort(reverse = True)

            for idx in dummy_ids:
                if g.nodes[idx]['smile'] == '&':
                    g.remove_node(idx)

        if show:          
           GTools.show_network_g_cnjtmol(g)

        return g


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
                    if 'idx_voc' in item.data:
                        if item.data['idx_voc'] == -1:
                            sml = item.data['smile']
                            if Chem.MolFromSmiles(sml) is not None:
                                new_vocs.append(sml)

                    bfs_ex_nodeid.append(item.idx)  
                    bfs_ex_vocid.append(item.data['idx_voc'] if 'idx_voc' in item.data else -1)
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
        except Exception as e:
            print('[CNJTMolTree.get_bfs_ex].Exception', e.args)
            return None, None, None, None, None

    def bfs_recon_validate(ctoken, bfs_ex_id = None, bfs_ex_smiles = None, show = False):
        g = None
        dummy_nodes = []

        if bfs_ex_id is None and bfs_ex_smiles is None:
            raise ValueError('[CNJTMol-bfs_ex_reconstruct]: input could not be None!')
            return g, dummy_nodes, bfs_ex_smiles, bfs_ex_id      
        
        nlen = len(bfs_ex_smiles)
        if nlen == 0 :
            return g, dummy_nodes, bfs_ex_smiles, bfs_ex_id

        try:
            for i in range(len(bfs_ex_smiles)):
                if not CNJMolUtil.is_dummy(bfs_ex_smiles[i]):                
                    bfs_ex_smiles[i] = CNJMolUtil.valid_smiles(bfs_ex_smiles[i], ctoken = ctoken)

            if nlen >  1 and bfs_ex_smiles[1] == ctoken.invalid_token:  #jump this dummy one
                bfs_ex_smiles.pop(1)
                #bfs_ex_smiles.remove(1)

            if bfs_ex_id is None:
                bfs_ex_id = []
                for sml in bfs_ex_smiles:
                    idx = ctoken.get_index(sml)
                    bfs_ex_id.append(ctoken.get_index(sml))
   
            if bfs_ex_smiles is None:
                bfs_ex_smiles = []
                for token_id in bfs_ex_id:
                    bfs_ex_smiles.append(ctoken.get_token(token_id))                        
           
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
        
            if show:
               GTools.show_network_g_cnjtmol(g)

        except Exception as e:
            print('[bfs_recon_validate Exception!],', e.args)
            g = None
            dummy_nodes = []

        return g, dummy_nodes, bfs_ex_smiles, bfs_ex_id

    def link_nodes(g):
        n_nodes = len(g.nodes)
        try:
            for i_s in range(n_nodes - 1):
                s_node = g.nodes[i_s]
                s_smt = s_node['smarts']
                attach_pos_s = set(re.findall(r"(\[\d+\*\]|!\[[^:]*:\d+\])", s_smt))
                if len(attach_pos_s) ==0:
                    continue

                for i_e in range(i_s + 1, n_nodes):  
                    e_node = g.nodes[i_e]
                    e_smt = e_node['smarts']
                    attach_pos_e = set(re.findall(r"(\[\d+\*\]|!\[[^:]*:\d+\])", e_smt))
                    if len(attach_pos_e) ==0:
                        continue

                    share_pos = attach_pos_s & attach_pos_e
                    if len(share_pos) > 0:
                        g.add_edge(i_s, i_e)
        except Exception as e:
            print('[bfs_ex_reconstruct_amt Exception link_nodes 1!],', e.args)
            return None
        
        if len(g.edges) == 0 and n_nodes > 1:
            try:
                for i_s in range(n_nodes - 1):
                    s_node = g.nodes[i_s]
                    s_smt = s_node['smarts']                 
                    s_smt = re.sub(r"(\[\d+\*\]|!\[[^:]*:\d+\])",r"[*]", s_smt)
                    attach_pos_s = set(re.findall(r"(\[\*\])", s_smt))
                    if len(attach_pos_s) ==0:
                        continue

                    for i_e in range(i_s + 1, n_nodes):  
                        e_node = g.nodes[i_e]
                        e_smt = e_node['smarts']
                        e_smt = re.sub(r"(\[\d+\*\]|!\[[^:]*:\d+\])",r"[*]", e_smt)
                        attach_pos_e = set(re.findall(r"(\[\*\])", e_smt))
                        if len(attach_pos_e) ==0:
                            continue

                        share_pos = attach_pos_s & attach_pos_e
                        if len(share_pos) > 0:
                            g.add_edge(i_s, i_e)
            except Exception as e:
                print('[bfs_ex_reconstruct_amt Exception link_nodes!],', e.args)
                return None 
         
        c = max(nx.weakly_connected_components(g), key=len)
        g = g.subgraph(c) 

        g = g.to_undirected()
        g = nx.minimum_spanning_tree(g)   

        return g

    def bfs_g_reorder(g):
        #to fix: '[3*]OC^[1*]C(=O)C1=C(C[4*])NC(C)=C(C#N)C1C1=CC=CC(C#N)=C1^[1*]OC^[2*]N[4*]^[2*]C(C)=O'  #cc
        #works fine, this is time consuming process, it could be ignored

        #1.000	0.625	0.584	0.984	0.865	0.934
        #1.000	0.632	0.589	0.984	0.871	0.932   reordered 

        n_idxs = list(g.nodes)
        sorted_nodes = []
        for idx in n_idxs:
            node = g.nodes[idx]
            n_atoms = len(node['smile'])
            sorted_nodes.append((idx, node, n_atoms))          

        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[2], reverse=True)

        g_sort = nx.Graph()
        edge_map = {}
        idx = 0
        s_nodes = []
        for snode in sorted_nodes:
            s_nodes.append(snode[1])
            g_sort.add_node(idx, 
                           smile    = snode[1]['smile'], 
                           clique   = snode[1]['clique'], 
                           nid      = snode[1]['nid'], 
                           is_leaf  = snode[1]['is_leaf'], 
                           idx_voc  = snode[1]['idx_voc'], 
                           jtnode   = snode[1]['jtnode'], 
                           smarts   = snode[1]['smarts'], 
                           )                
            edge_map[snode[0]] = idx
            idx += 1

        for (s,e) in g.edges:
            s_s = edge_map[s]
            s_e = edge_map[e]
            g_sort.add_edge(s_s, s_e)

        return g_sort


    def bfs_ex_reconstruct_amt(ctoken, bfs_ex_id = None, bfs_ex_smiles = None, clean_up = True, show = False):

        g, dummy_nodes, bfs_ex_smiles, bfs_ex_id =  CNJTMolTree.bfs_recon_validate(ctoken, bfs_ex_id = bfs_ex_id, bfs_ex_smiles = bfs_ex_smiles, show = show)      
         
        try:
            g = CNJTMolTree.link_nodes(g)
            
            if show:
                GTools.show_network_g_cnjtmol(g)
            
            g = CNJTMolTree.bfs_g_reorder(g)  #this is time consuming process, it could be ignored
            if show:
                GTools.show_network_g_cnjtmol(g)
        except Exception as e:
            print('[bfs_ex_reconstruct_amt Exception clean_up!],', e.args)
            return None, None, None          

        n_nodes = len(g.nodes)

        bfs_node_map = {}

        idx = 0
        bfs_tree = CNJTMolTreeNode(idx = idx, nx_node = g.nodes[idx])  #the first node
        bfs_node_map[idx] = bfs_tree
        idx+=1
        while idx < n_nodes:
            node = CNJTMolTreeNode(idx = idx, nx_node = g.nodes[idx])
            bfs_node_map[idx] = node
            idx+=1

        return bfs_tree, g, bfs_node_map

    def bfs_ex_reconstruct(ctoken, bfs_ex_id = None, bfs_ex_smiles = None, clean_up = True, show = False):
        if bfs_ex_id is None and bfs_ex_smiles is None:
            raise ValueError('[CNJTMol-bfs_ex_reconstruct]: input could not be None!')
            return None, None, None        
        
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
  
        skt_wrong = 0
        if clean_up:
            edged_ids = set(edged_ids)
            all_ids = set([*range(0, len(g.nodes), 1)])
            dummy_ids = list(all_ids - edged_ids)

            dummy_ids.sort(reverse = True)
            for idx in dummy_ids:
                if g.nodes[idx]['smile'] != '&' and g.nodes[idx]['smile'] != '^':
                    skt_wrong += 1                   
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

        if skt_wrong> 0:
           print('!![skt wrong]:', bfs_ex_smiles)
          
        return bfs_tree, g, bfs_node_map, skt_wrong

    def reorder_bfs_tree(bfs_binary_tree, id_map):
        queue = []
        point = bfs_binary_tree
        queue.append(point)
       

        while len(queue) != 0:
            item = queue.pop(0)

            if item is not None:
                queue.append(item.left)
                queue.append(item.right)

            item.idx = id_map[item.idx]

        return 

    def get_bfs_nodes(self):
        bfs_idkey = GTools.BFS(self.graph_nx, source = 0)
             
        bfs_node_list = [] 
        bgs_mt_idx = []
        bfs_smiles = []

        for i in range(len(bfs_idkey)):
            idx = bfs_idkey[i]
            bfs_node_list.append(self.graph_nx.nodes[idx]['jtnode'])  #not match
            bfs_smiles.append(bfs_node_list[-1].smiles)

        return bfs_node_list, bfs_smiles

    def convert_to_smiles(ctoken, bfs_binary_tree):
        smile = ''

        return smile

    def get_node_by_nid(self, nid):
        for node in self.nodes:
            if node.nid == nid:
                return node
        return None


    def get_nb_map(g):
        edges = list(g.edges)
        n_nodes = len(g.nodes)
        neighbor_map = [()] * n_nodes   #self.neighbor_map	[{1}, {0, 2}, {1, 3, 5}, {2, 4}, {3}, {2, 6}, {5}]	list

        for (s,e) in edges:
            sbm = set(neighbor_map[s])
            sbm.add(e)
            neighbor_map[s] = sbm

            sbm = set(neighbor_map[e])
            sbm.add(s)
            neighbor_map[e] = sbm

        return neighbor_map

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
        self.neighbor_map = CNJTMolTree.get_nb_map(span_g)

        #self.neighbor_map = [()] * len(self.nodes)   #self.neighbor_map	[{1}, {0, 2}, {1, 3, 5}, {2, 4}, {3}, {2, 6}, {5}]	list
        #for (s,e) in self.edges:
        #    sbm = set(self.neighbor_map[s])
        #    sbm.add(e)
        #    self.neighbor_map[s] = sbm

        #    sbm = set(self.neighbor_map[e])
        #    sbm.add(s)
        #    self.neighbor_map[e] = sbm

        for i, node in enumerate(self.nodes):
            nbm = self.neighbor_map[i]      
            node.neighbors = []
            for nid in nbm:
                node.neighbors.append(self.nodes[nid])

            node.is_leaf = (len(node.neighbors) == 1)        

        return span_g      


    def convert_to_nx_binarytree(self, show = False):
        bt = nx.DiGraph()
        n_nodes = len(self.nodes)
        #g.add_nodes_from(i for i in range(n_nodes))

        for i, node in enumerate(self.nodes):
            bt.add_node(i, 
                        smile = node.smiles, 
                        clique = node.clique, 
                        nid = node.nid,  #this is 1 based, zero for end node
                        is_leaf = node.is_leaf,
                        #idx_voc = self.jtvoc.get_index(node.smiles),
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
                bt.add_edge(start, end)
                nbs.append(end)

            self.neighbor_map.append(nbs)

        if show:
            GTools.show_network_g_cnjtmol(bt)

        return g


class CNJMolUtils:
    def encode_single(smls, ctoken, dec_alg):
        #print('[smls is]:', smls)
        try:
            sub_smils = smls.strip().split('.')  #to seperate molecules for reaction
            combine_sml = []
            combine_smt = []
            amt_bfs_smarts = ''

            for sml in sub_smils:
                cnjtmol = CNJTMolTree(sml, ctoken = ctoken, dec_alg = dec_alg)
                if cnjtmol is  None:
                    print('cnjtmol is None')

                amt_bfs_smarts = cnjtmol.amt_bfs_smarts
     
                bfs_ex_smiles = cnjtmol.bfs_ex_smiles
                #print('bfs_ex_smiles =', bfs_ex_smiles)

                bfs_ex_smarts = cnjtmol.bfs_ex_smarts
                #print('bfs_ex_smarts =', bfs_ex_smarts)

                atom_env = cnjtmol.atom_env
                s, skeleton = CNJMolUtil.combine_ex_smiles(bfs_ex_smiles)
                combine_sml.append(s)
     
                s, skeleton = CNJMolUtil.combine_ex_smiles(bfs_ex_smarts)
                combine_smt.append(s)

                words = CNJMolUtil.split_ex_smiles(s)
                #print('words=',words)

            combine_sml = '.'.join(s for s in combine_sml)
            #print('combine_ex_smiles = ', s)
  
            combine_smt = '.'.join(s for s in combine_smt)
            #print('combine_ex_smarts = ', s)
        except Exception as e:
            print('[CNJMolUtils.encode_single].exception:', e.args)
            combine_sml = 'CC&&&'
            combine_smt = 'CC&&&'

        return combine_sml, combine_smt, amt_bfs_smarts


def preprocess():
    

    dec_algs = [
                #Fragment_Alg.Vanilla,

                #Fragment_Alg.JTVAE,
                #Fragment_Alg.BRICS,
                #Fragment_Alg.BRICS_Base,
                #Fragment_Alg.MMPA,
                #Fragment_Alg.Scaffold,

                #Fragment_Alg.BRICS_DY,
                Fragment_Alg.MMPA_DY,
                #Fragment_Alg.Scaffold_DY,
                #Fragment_Alg.RBrics_DY,
                ]


    #kekuleSmiles = False
    kekuleSmiles = True
  
    ctoken = CTokens(STDTokens_Frag_File(None), max_length = 256,   invalid = True, onehot = False)

    #smlfile = r'../RawData/Chembl/Test/Chembl_test.smi'
    smlfile = r'../RawData/Example/mol.smi'

    print(smlfile)
    df = pd.read_csv(smlfile, squeeze=True, delimiter=',',header = None, skip_blank_lines = True) 
    smiles_list = list(df.values)

    for dec_alg in dec_algs:
        bfs_ex_list = []
        bfs_smile_list = []
        bfs_smile_list_join = []
        bfs_smarts_list_join = []
        bfs_amt_list_join = []
        idd_amt_list_join = []
        org_smile_list = []
        bfs_max_Len = 0
        vocab_list = set()
        skt_list = []
        atom_env_list = []

        save_prex = dec_alg.name

        for i, sml in tqdm(enumerate(smiles_list), total = len(smiles_list),  desc = 'parsing smiles ...'):
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
                joined_sub_amt = ''
                joined_sub_idd = ''
                skeleton_sub = ''

                joined_smiles = None
                for i, sub_s in enumerate(sub_smils):
                    cnjtmol = CNJTMolTree(sub_s, ctoken = ctoken, dec_alg = dec_alg) 

                    if cnjtmol.mol is not None:
                        for c in cnjtmol.nodes:
                            vocab_list.add(c.smiles)

                        joined_amt  = cnjtmol.amt_bfs_smarts  #tsis
                        joined_idd  = cnjtmol.amt_dfs_smarts

                        joined_smiles, skeleton = CNJMolUtil.combine_ex_smiles(cnjtmol.bfs_ex_smiles)
                        joined_smarts, _        = CNJMolUtil.combine_ex_smiles(cnjtmol.bfs_ex_smarts)
  
                        if i > 0:
                            org_smile_sub   += '.'
                            bfs_smile_list_sub.extend(['.'])
                            bfs_smart_list_sub.extend(['.'])
                            joined_sub_smiles   += '.'
                            joined_sub_smarts   += '.'
                            skeleton_sub        += '.'
                            joined_sub_amt      += '.'
                            joined_sub_idd      += '.'

                        org_smile_sub = org_smile_sub + sub_s
                        bfs_smile_list_sub.extend(cnjtmol.bfs_ex_smiles)
                        bfs_smart_list_sub.extend(cnjtmol.bfs_ex_smarts)

                        joined_sub_smiles = joined_sub_smiles + joined_smiles
                        joined_sub_smarts = joined_sub_smarts + joined_smarts
                        joined_sub_amt    = joined_sub_amt + joined_amt
                        joined_sub_idd    = joined_sub_idd + joined_idd

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
                bfs_amt_list_join.append(joined_sub_amt)
                idd_amt_list_join.append(joined_sub_idd)

                skt_list.append(skeleton_sub)
                
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

        max_len = len(max(skt_list))
        output = smlfile + f'.[{save_prex}][{bfs_max_Len}]_skt_lst[{max_len}].csv'
        df = pd.DataFrame(skt_list)
        df.to_csv(output, index = False, header=False, na_rep="NULL")
        print('[skt_list]:',len(skt_list))
        skt_list = list(set(skt_list))
        skt_list.sort()
        output = smlfile + f'.[{save_prex}][{bfs_max_Len}]_skt_set[{max_len}].csv'
        df = pd.DataFrame(skt_list)
        df.to_csv(output, index = False, header=False, na_rep="NULL")
        print('[skt_set]:',len(skt_list))

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

            output = smlfile + f'.[{save_prex}][{bfs_max_Len}]_TSIS.csv'  #TSIS
            df = pd.DataFrame(bfs_amt_list_join)
            df.to_csv(output, index = False, header=False, na_rep="NULL")

            output = smlfile + f'.[{save_prex}][{bfs_max_Len}]_TSISD.csv'
            df = pd.DataFrame(idd_amt_list_join)
            df.to_csv(output, index = False, header=False, na_rep="NULL")

            #----------------------------
            TSIO = []
            TSIR = []
            for s in bfs_amt_list_join:
                frags = s.split('^')
                tsio = sorted(frags, key=len, reverse=True)            
                random.shuffle(frags)

                tsio = '^'.join(tsio)
                tsir = '^'.join(frags)
                TSIO.append(tsio)
                TSIR.append(tsir)

            output = smlfile + f'.[{save_prex}][{bfs_max_Len}]_TSISO.csv'
            df = pd.DataFrame(TSIO)
            df.to_csv(output, index = False, header=False, na_rep="NULL")
        
            output = smlfile + f'.[{save_prex}][{bfs_max_Len}]_TSISR.csv'
            df = pd.DataFrame(TSIR)
            df.to_csv(output, index = False, header=False, na_rep="NULL")

        else: #Fragment_Alg.Vanilla
            output = smlfile + f'.[{save_prex}][{bfs_max_Len}]_TSV.csv'
            df = pd.DataFrame(bfs_smile_list_join)
            df.to_csv(output, index = False, header=False, na_rep="NULL")

    return

def test_encode():
    smls = 'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'  #celecoxib

    dec_algs = [
         #Fragment_Alg.Vanilla,

         #Fragment_Alg.JTVAE,
         #Fragment_Alg.BRICS,
         #Fragment_Alg.BRICS_Base,
         #Fragment_Alg.MMPA,
         #Fragment_Alg.Scaffold,

         #Fragment_Alg.BRICS_DY,
         Fragment_Alg.MMPA_DY,
         #Fragment_Alg.Scaffold_DY,

         #Fragment_Alg.RBrics_DY,
        ]

    print('[smls is]:', smls)

    ctoken = CTokens(STDTokens_Frag_File(None), max_length = 256, invalid = True, onehot = False)


    for dec_alg in dec_algs:
        combine_sml, combine_smt, amt_bfs_smarts = CNJMolUtils.encode_single(smls, ctoken, dec_alg)
    
        print('[dec_alg is]:', dec_alg.name)
        print('[TSSA/TSDY]:', combine_sml)  
        print('[TSID     ]:', combine_smt)     
        print('[TSIS     ]:', amt_bfs_smarts)     
   
    return 



if __name__ == '__main__':

    #test_encode()


    preprocess()

