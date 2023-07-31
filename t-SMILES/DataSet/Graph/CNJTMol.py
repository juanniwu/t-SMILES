
import numpy as np
import pandas as pd
from tqdm import tqdm

import networkx as nx

import rdkit.Chem as Chem

from Tools.MathUtils import BCMathUtils
from Tools.GraphTools import GTools

from DataSet.JTNN.MolTree import Vocab, MolTree
from DataSet.STDTokens import CTokens, STDTokens_Frag_File

from DataSet.Graph.CNJMolUtil import CNJMolUtil
from DataSet.Graph.CNXBinaryTree import CNXBinaryTree

from MolUtils.RDKUtils.Frag.RDKFragUtil import Fragment_Alg


           
class CNJTMolTreeNode():
    disconnect_char = '^'  #'.' is used in SMILES, '~'is used in SMARTS
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

    #def 
#-----------------------------------------------------------------
     

class CNJTMolTree(MolTree):        
    def __init__(self, smiles,
                 jtvoc = None, #: Vocab
                 ctoken = None, #: CTokens, 
                 dec_alg = Fragment_Alg.Scaffold,
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
                        smiles:str,  #one molecule
                        jtvoc: Vocab,
                        ctoken: CTokens,
                        dec_alg = 'BRICS',
                        kekuleSmiles = True,
                        ):
        try:
            super(CNJTMolTree, self).__init__(smiles = smiles, dec_alg = dec_alg, kekuleSmiles = kekuleSmiles)

            self.n_nodes = len(self.nodes)

            self.neighbor_map = []
            self.nx_parent_map = {}
            self.nx_child_map = {}
            self.nx_parent_map[0] = 0

            self.graph_nx = self.convert_to_nx(show = True)

            self.bfs_idx = GTools.BFS(self.graph_nx)    #Breadth first         
            self.dfs_idx = GTools.DFS(self.graph_nx )   #Depth-First Search


            self.build_relation(self.bfs_idx)

            self.bfs_binary_tree = self.create_bfs_binary_tree_ex(show = True)

            #self.bfs_binary_tree = self.create_bfs_binary_tree(show = True)
            self.bfs_ex_nodeid, self.bfs_ex_vocids, self.bfs_ex_smiles, self.new_vocs = CNJTMolTree.get_bfs_ex(self.ctoken, 
                                                                                                               self.bfs_binary_tree, 
                                                                                                               extra_dummy = True)   #generate advanced bfs 

            self.bfs_idx_ex = GTools.BFS(self.nx_binarytree)    #Breadth first         
            self.dfs_idx_ex = GTools.DFS(self.nx_binarytree )   #Depth-First Search

            if self.bfs_ex_vocids is None:
                self.mol = None
            self.combine_ex_smiles, self.skeleton = CNJMolUtil.combine_ex_smiles(self.bfs_ex_smiles)

        except Exception as e:
            print('init_from_smile Exception:', smiles)
            print(e.args)
            self.mol = None
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

        if last_visited != -1:
             brother = self.bfs_node_map[last_visited]

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
            parent = self.bfs_node_map[idx]  #parent is itself

        self.nx_parent_map[idx] = parent.idx   #
        return parent

    def create_bfs_binary_tree_ex(self, show = False):
        #self.nx_parent_map = {}
        #self.nx_child_map = {}
 
        self.nx_binarytree = CNXBinaryTree()  #DiGraph
         
        visited = np.zeros((self.n_nodes))

        self.bfs_idx_ex = []
        self.bfs_node_map = {}   #not the same as self.bfs_idx
    
        idx = 0
        nx_node = self.graph_nx.nodes[idx]
        self.bfs_binary_tree = CNJTMolTreeNode(idx=idx, nx_node=nx_node)  #the first node
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
                                        data  = bfs_node
                                        )              

        for idx in self.nx_child_map:
            nbs = self.nx_child_map[idx]
            if len(nbs) == 0: #no child
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
                    self.nx_binarytree.add_edge_right(point.idx, b_idx)  #nx_binarytree left

                    brother = self.nx_binarytree.nodes[b_idx]['data']
                    point = point.add_right(brother) 

                    visited[b_idx]  = 1
        #if show:          
        #   GTools.show_network_g_cnjtmol(self.nx_binarytree)

        self.nx_binarytree.make_full()

        for idx in self.bfs_idx:
            self.bfs_node_map[idx] = self.nx_binarytree.nodes[idx]['data']

        #if show:          
        #   GTools.show_network_g_cnjtmol(self.nx_binarytree)

        self.bfs_binary_tree = self.nx_binarytree.nodes[0]['data']
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
                            data  = item
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

        #if show:          
        #   GTools.show_network_g_cnjtmol(g)

        return g


    def create_bfs_binary_tree(self, show=False):  #
        #input  : self.bfs_idx which comes from self.graph_nx which comes from CNJTMolTree
        #output : self.nx_binarytree which is a nx.DiGraph
        #       : self.bfs_node_map which map idx to CNJTMolTreeNode

        self.bfs_idx_ex = []
        self.nx_binarytree = CNXBinaryTree()

        visited = np.zeros((self.n_nodes))
        self.bfs_node_map = {}   #not the same as self.bfs_idx

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
                                       nid  = 0
                                        )

        for i in range(0, len(self.bfs_idx)) :
            idx = self.bfs_idx[i]
            neighbors = self.neighbor_map[idx]

            if visited[idx] == 1 and len(neighbors) <= 2:  # = 2: two neighbors, one is parent, one is child, if 1 then only a parent
                #point = self.bfs_node_map[idx]
                continue

            #if visited[idx] == 0:
            parent = self.find_parent_node(idx, visited)
            if parent.idx != idx:
                nx_node = self.graph_nx.nodes[idx]
                bfs_node = CNJTMolTreeNode(idx=idx, nx_node=nx_node)
                point = parent.add_left(bfs_node)                            
                self.bfs_node_map[idx] = bfs_node
                visited[idx] = 1

                self.nx_binarytree.add_edge_left(parent.idx, bfs_node.idx)  #nx_binarytree left

            first = True
            for nb in neighbors:
                if visited[nb] == 1:
                    #point = self.bfs_node_map[nb]
                    continue

                visited[nb] = 1
                nx_node = self.graph_nx.nodes[nb]
                bfs_node = CNJTMolTreeNode(idx=nb, nx_node=nx_node)
                self.bfs_node_map[nb] = bfs_node

                if first:
                    nbpoint = point.add_left(bfs_node)   #child
                    first = False
                    #point = nbpoint

                    self.nx_binarytree.add_edge_left(point.idx, bfs_node.idx)  #nx_binarytree left

                else:
                    self.nx_binarytree.add_edge_right(nbpoint.idx, bfs_node.idx)  #nx_binarytree left
                    
                    nbpoint = nbpoint.add_right(bfs_node)  #brother in JTVAE  
                                    

        #update binarytree
        for node in self.nx_binarytree.nodes:
            idx = node
            if isinstance(idx, CNJTMolTreeNode):
                continue
            else:
                self.nx_binarytree.nodes[idx]['nid'] = idx
                self.nx_binarytree.nodes[idx]['smile'] = self.bfs_node_map[idx].data['smile']

        self.nx_binarytree.make_full()

        #if show:          
        #   GTools.show_network_g_cnjtmol(self.nx_binarytree)
                    
        return self.bfs_binary_tree

    def get_bfs_ex(ctoken, bfs_binary_tree, extra_dummy = True): #should be get_bfs_ex
        #get bfs list from full binary tree using queue 
        #
        #bfs_idx= [0, 11, 1, 6, 8, 10, 2, 3, 4, 7, 9, 5]
        queue = []

        point = bfs_binary_tree
        queue.append(point)
       
        bfs_ex_nodeid = []
        bfs_ex_vocid = []
        bfs_ex_smiles = []
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
                    #    return None, None, None

                    bfs_ex_nodeid.append(item.idx)  #to verify 
                    bfs_ex_vocid.append(item.data['idx_voc'])
                    bfs_ex_smiles.append(item.data['smile'])
                else:
                    #add a dummy:
                    bfs_ex_nodeid.append('p')
                    #bfs_ex_vocids.append(CNJTMolTree.dummy_node['idx_voc'])
                    #bfs_ex_smiles.append(CNJTMolTree.dummy_node['smile'])
                    bfs_ex_vocid.append(ctoken.invalid_index)
                    bfs_ex_smiles.append(ctoken.invalid_token)

                queue.pop(0)
            #end while
            
            if extra_dummy:
                bfs_ex_nodeid.insert(1, 'p')
                bfs_ex_vocid.insert(1, ctoken.invalid_index)
                bfs_ex_smiles.insert(1, ctoken.invalid_token)

            return bfs_ex_nodeid, bfs_ex_vocid, bfs_ex_smiles, new_vocs
        except:
            return None, None, None, None

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
                       jtnode = None
                       )
                       
            if bfs_ex_smiles[i]  == '&':
                dummy_nodes.append(i)

        bfs_node_map = {}

        queue = []
        idx = 0
        t_id = bfs_ex_id[0]
        sml = bfs_ex_smiles[0]
        bfs_tree = CNJTMolTreeNode(idx = idx, nx_node = g.nodes[idx])  #the first node
        bfs_node_map[idx] = bfs_tree
        idx+=1

        edged_ids.append(0)
        #parent = bfs_tree

        if nlen == 1 or nlen == 2:
            return bfs_tree, g, bfs_node_map
   
        queue.append(bfs_tree)

        t_id_1 = bfs_ex_id[1]
        sml_1 = bfs_ex_smiles[1]
        if sml_1 == ctoken.invalid_token:  #jump this dummy one
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

                node_left = CNJTMolTreeNode(idx = idx, nx_node=g.nodes[idx])  #the first node
                bfs_node_map[idx] = node_left
                idx+=1
                node = parent.add_left(node_left)
                g.add_edge_left(parent.idx, node.idx)  #nx_binarytree left
                edged_ids.append(parent.idx)
                edged_ids.append(node.idx)

                #-------------------
                node_right = CNJTMolTreeNode(idx = idx, nx_node=g.nodes[idx])  #the first node                           
                bfs_node_map[idx] = node_right
                idx+=1
                node = parent.add_right(node_right)
                g.add_edge_right(parent.idx, node.idx)  #nx_binarytree left
                edged_ids.append(parent.idx)
                edged_ids.append(node.idx)

                #if sml_left != ctoken.invalid_token:#'&':  #this is dummy_token 
                if tid_left != ctoken.invalid_index:#'&':  #this is dummy_token 
                    queue.append(node_left)
                if tid_right != ctoken.invalid_index:#'&':  #this is dummy_token 
                    queue.append(node_right)                

        except Exception as e:
            print('[bfs_ex_reconstruct Exception!]-ignore!')
            print(e.args)
  
        #if show:          
        #   GTools.show_network_g_cnjtmol(g)

        #clean up
        if clean_up:
            edged_ids = set(edged_ids)
            all_ids = set([*range(0, len(g.nodes), 1)])
            dummy_ids = list(all_ids - edged_ids)

            dummy_ids.sort(reverse = True)
            for idx in dummy_ids:
                    g.remove_node(idx)

        #reorder ids:
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

        #if show:
        #   GTools.show_network_g_cnjtmol(g)
          
        return bfs_tree, g, bfs_node_map

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


    def convert_to_smiles(ctoken, bfs_binary_tree):
        smile = ''

        return smile

    def get_node_by_nid(self, nid):
        for node in self.nodes:
            if node.nid == nid:
                return node
        return None

    def convert_to_nx(self, show = False):
        g = nx.Graph()
        n_nodes = len(self.nodes)

        for i, node in enumerate(self.nodes):
            g.add_node(i, 
                       smile = node.smiles, 
                       clique = node.clique, 
                       nid = node.nid,  #this is 1 based, zero for end node
                       is_leaf = node.is_leaf,
                       idx_voc = self.ctoken.get_index(node.smiles),
                       jtnode = node
                       )

        for i, node in enumerate(self.nodes):
            start = node.nid - 1
            nblist = node
            nbs = []
            for nb in node.neighbors:
                end = nb.nid - 1
                g.add_edge(start, end)
                nbs.append(end)

            self.neighbor_map.append(nbs)

        #T = nx.minimum_spanning_tree(g)

        #if show:
        #    GTools.show_network_g_cnjtmol(g)

        return g

    def convert_to_nx_binarytree(self, show = False):
        bt = nx.DiGraph()
        n_nodes = len(self.nodes)

        for i, node in enumerate(self.nodes):
            bt.add_node(i, smile = node.smiles, 
                       clique = node.clique, 
                       nid = node.nid,  #this is 1 based, zero for end node
                       is_leaf = node.is_leaf,
                       idx_voc = self.ctoken.get_index(node.smiles),
                       jtnode = node
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

        #if show:
        #    GTools.show_network_g_cnjtmol(bt)

        return g

    def onehot_code(inputx, ctoken, maxlen, start_end = True, pad = True):
        if start_end:
            inputx.insert(0, ctoken.start_index)
            inputx.append(ctoken.end_index)
      
        nlen = len(inputx)

        if pad:
            if nlen > maxlen:
                raise ValueError('get_oh_code error')

            #inputx = np.pad(inputx, (0, maxlen - len(bfs_ex)), mode='constant', constant_values=CNJTMolTree.PAD_INDEX)
            inputx = np.pad(inputx, (0, maxlen - len(inputx)), mode='constant', constant_values=ctoken.pad_index)

        ohcode = []
        for s in inputx:
            oh = np.zeros(ctoken.n_tokens, dtype=np.int32)
            oh[int(s)] = 1
            ohcode.append(oh)

        return np.array(ohcode)
        
    def onehot_decode(ohcode):
        nlen = len(ohcode)
        bfs_ex = []

        for oh in ohcode:
            pos = np.argmax(oh)
            bfs_ex.append(pos)

        return bfs_ex

    def onehot_decode_token(ohcode, ctoken):
        nlen = len(ohcode)
        bfs_ex_token = []

        for oh in ohcode:
            pos = np.argmax(oh)
            
            tk = ctoken.get_token(pos)
            bfs_ex_token.append(tk)

        return bfs_ex_token

    def dfs_jt_nx(jtmoltree: MolTree,):
        return

    def read_dfc_csv(filename, ctoken, pad = True, oh = True):
        data = pd.read_csv(filename, header = None).values

        max_len = 0
        dfs_list = []
        for item in tqdm(data, total = len(data)):
            error = False
            code = []
            for d in item:
                if str(d) != 'nan':  #pandas 'nan'
                    d = int(d)
                    if d == -1:
                       error = True
                       break
                    else:
                        code.append(d)
                else:
                    break

            if not error:
                dfs_list.append(code)
                if len(code)> max_len:
                    max_len = len(code)

        print(f'====Reading file done: max_len + 4  = {max_len + 4}, total = {len(dfs_list)}')   

        xcode = []
        ycode = []
        max_len =  max_len + 4  #4 means: ' ', '<', '>', '&'
        if oh:
            for i in range(len(dfs_list)):
                bfs_ex = dfs_list[i]
                ohcode = CNJTMolTree.onehot_code(ctoken = ctoken, inputx = bfs_ex, maxlen = max_len, start_end = True, pad = True)
                x = ohcode[0:-1,:]
                y =  ohcode[1:,:]
                xcode.append(x)
                ycode.append(y)

        else:
            for i in range(len(dfs_list)):
                bfs_ex = dfs_list[i]
                if pad:
                    bfs_ex = np.pad(bfs_ex, (0, max_len - len(bfs_ex)), mode='constant', constant_values = ctoken.pad_index)

                x = bfs_ex[0:-1]
                y = bfs_ex[1:]
                xcode.append(x)
                ycode.append(y)
          
        return np.array(xcode), np.array(ycode), max_len


def test():
    smls = 'CC(C)CC1=C(CC(C)C)[AlH2+]1.CCCCCCCCCCCCCCCCNc1ccc(C#N)c(F)c1.CO.Cc1ccccc1.[H-]'

    #Vanilla    = CC(C)CC1=C(CC(C)C)[Al+]1&CCCCCCCCCCCCCCCCNC1=CC=C(C#N)C(F)=C1&CO&CC1=CC=CC=C1&[HH2-]&&&    
    #JTVAE      = CC&C&CC&&CC^CC&C1=C[Al+]1&CC&CC&C&CC&&CC&CC^CC&CC&CC&CC&CC&CC&CC&CC&CC&CC&CC&CC&CC&CC&CN&CN&C1=CC=CC=C1&CC&C#N^CF&&&CO^CC&C1=CC=CC=C1&[HH2-]&&&
    #BRICS      = CC&C&CC&&CC^CC&C1=C[Al+]1&CC&CC&C&CC&&CC&C1=CC=CC=C1^CN&CN^CC^CC&C#N^CF^CC&&&CO&CC&CC&CC&C1=CC=CC=C1&CC&[HH2-]&CC&&&CC&CC&CC&CC&CC&CC&CC&CC&CC&&&
    #BRICS_Base = CC(C)CC1=C(CC(C)C)[Al+]1&CCCCCCCCCCCCCCCC&CN&N#CC1=CC=C(N)C=C1F&&CO^CC1=CC=CC=C1&[HH2-]&&&
    #MMPA       = CC&C&CC&&CC^CC^C1=CC=CC=C1^CC(C)CC1=C[Al+]1&CN&&&CN^CC^CC&C#N^CF^CC&&&CO&CC&C1=CC=CC=C1&CC&CC&CC&[HH2-]&CC&&&CC&CC&CC&CC&CC&CC&CC&CC&CC&&&
    #Scaffold   = CC(C)C&CC&C1=C[Al+]1&CC&CC(C)C&&CCCCCCCCCCCCCCCCN^CN&C1=CC=CC=C1&CC&C#N^CF&&CO&C1=CC=CC=C1&CC&[HH2-]&&&
    #========================================================
    
    dec_alg = Fragment_Alg.Vanilla
    #dec_alg = Fragment_Alg.JTVAE
    #dec_alg = Fragment_Alg.BRICS
    #dec_alg = Fragment_Alg.BRICS_Base
    #dec_alg = Fragment_Alg.MMPA
    #dec_alg = Fragment_Alg.Scaffold


    print('[seed is]:', smls)
    print('[dec_alg is]:', dec_alg.name)
    #------------------------------------------------------
    maxlen = 256
    vocab_file = r'H:\GitHub\t-SMILES\RawData\QM9\JTVAE\QM9.smi.[JTVAE][24]_token.voc.smi'

    ctoken = CTokens(STDTokens_Frag_File(vocab_file), is_pad = True, pad_symbol = ' ', startend = True,
                     max_length = maxlen,  flip = False, invalid = True, onehot = False)

     #----------    
    cnjtmol = CNJTMolTree(smls, ctoken = ctoken, dec_alg = dec_alg)
    if cnjtmol is  None:
        print('cnjtmol is None')
     
    bfs_ex_smiles = cnjtmol.bfs_ex_smiles
    print('bfs_ex_smiles=', bfs_ex_smiles)

    s, skeleton = CNJMolUtil.combine_ex_smiles(bfs_ex_smiles)
    words = CNJMolUtil.split_ex_smiles(s)
    print('combine_ex_smiles=', s)
    print('words=',words)

    return 


def preprocess():

    maxlen = 256
    vocab_file = r'H:\GitHub\t-SMILES\RawData\AID1706\active\Scaffold\active_smiles.smi.[Scaffold][24]_token.voc.smi'

    ctoken = CTokens(STDTokens_Frag_File(vocab_file), is_pad = True, pad_symbol = ' ', startend = True,
                     max_length = maxlen,  flip = False, invalid = True, onehot = False)
    #-----------------------------------------------------------------------------------
    smlfile = r'H:\GitHub\t-SMILES\RawData\AID1706\active.smi'

    #------------------------------------------------------   
    #dec_algs = [Fragment_Alg.Vanilla]
    #dec_algs = [Fragment_Alg.JTVAE]
    #dec_algs = [Fragment_Alg.BRICS]
    #dec_algs = [Fragment_Alg.BRICS_Base]
    #dec_algs = [Fragment_Alg.MMPA]
    dec_algs = [Fragment_Alg.Scaffold]

    bfs_smile_list = []
    bfs_smile_list_join = []
    org_smile_list = []
    bfs_max_Len = 0
    vocab_list = set()
    skt_list = set()
    
    df = pd.read_csv(smlfile, squeeze=True, delimiter=',',header = None) 
    smiles_list = list(df.values)

    for dec_alg in dec_algs:
        print(f'----------[{dec_alg.name}]-------------')
        save_prex = dec_alg.name

        for i, sml in tqdm(enumerate(smiles_list), total = len(smiles_list),  desc = 'parsing smiles ...'):
            if sml is None or str(sml) == 'nan' or len(sml) > maxlen:
                continue
        
            try:
                cnjtmol = CNJTMolTree(sml, ctoken = ctoken, dec_alg = dec_alg)

                if cnjtmol.mol is not None:
                    for c in cnjtmol.nodes:
                        vocab_list.add(c.smiles)

                    org_smile_list.append(sml)

                    bfs_smile_list.append(cnjtmol.bfs_ex_smiles)            

                    smil = list(filter(None, cnjtmol.bfs_ex_smiles))
                    joined, skeleton = CNJMolUtil.combine_ex_smiles(cnjtmol.bfs_ex_smiles)

                    bfs_smile_list_join.append(joined)
                    skt_list.add(skeleton)
                
                    if len(cnjtmol.bfs_ex_vocids) > bfs_max_Len:
                        bfs_max_Len = len(cnjtmol.bfs_ex_smiles)

            except:
                print('preprocess exception:', sml)
                continue    

       
    ctoken.maxlen = bfs_max_Len + 4  #
       
    vocab = sorted(vocab_list)
    output = smlfile + f'.[{save_prex}][{bfs_max_Len}]_token.voc.smi'
    df = pd.DataFrame(vocab)
    df.to_csv(output, index = False, header=False, na_rep="NULL")

    #output = smlfile + f'.bfs[{bfs_max_Len}]_org.csv'          #
    output = smlfile + f'.[{save_prex}][{bfs_max_Len}]_org.csv'
    df = pd.DataFrame(org_smile_list)
    df.to_csv(output, index = False, header=False, na_rep="NULL")

    output = smlfile + f'.[{save_prex}][{bfs_max_Len}]_join.csv'
    df = pd.DataFrame(bfs_smile_list_join)
    df.to_csv(output, index = False, header=False, na_rep="NULL")
 
    skt_list = list(skt_list)
    skt_list.sort()
    output = smlfile + f'.[{save_prex}][{bfs_max_Len}]_skt.csv'
    df = pd.DataFrame(skt_list)
    df.to_csv(output, index = False, header=False, na_rep="NULL")
    print('[skt_list]:',len(skt_list))

    return

if __name__ == '__main__':

    test()

    #preprocess()



