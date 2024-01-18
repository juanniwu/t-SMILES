import networkx as nx
import matplotlib.pyplot as plt

from Tools.GraphTools import GTools

class CNXBinaryTree(nx.DiGraph):
    def __init__(self, **attr):
        super(CNXBinaryTree, self).__init__(attr)
        return

    def add_left():
        #edge['label'] = 'L'
        return 

    def add_right():
        #edge['label'] = 'R'
        return

    def add_edge_left(self, u_of_edge, v_of_edge, **attr):
        self.add_edge(u_of_edge, v_of_edge, label = 'L')
        return 

    def add_edge_right(self, u_of_edge, v_of_edge, **attr):
        self.add_edge(u_of_edge, v_of_edge, label = 'R')
        return 

    def make_full(self):
        nodes = (list(self.nodes)).copy()
        n_nodes = len(self.nodes)
        dummy_node_id = n_nodes

        for node in nodes:
            edges_list = self.get_edge_by_start_node(node) 
            if len(edges_list) == 2:
                continue
            elif len(edges_list) == 1:
                edge = edges_list[0]
                edata = self.get_edge_data(*edge)

                if edata['label'] == 'L':
                    self.add_edge_right(node, dummy_node_id)
                    self.nodes[dummy_node_id]['smile'] = '&'
                    self.nodes[dummy_node_id]['smarts'] = '&'
                    dummy_node_id += 1 
                elif edata['label'] == 'R':
                    self.add_edge_left(node, dummy_node_id)                               
                    self.nodes[dummy_node_id]['smile'] = '&'
                    self.nodes[dummy_node_id]['smarts'] = '&'
                    dummy_node_id += 1 

            elif len(edges_list) == 0:
                self.add_edge_right(node, dummy_node_id)
                self.nodes[dummy_node_id]['smile'] = '&'
                self.nodes[dummy_node_id]['smarts'] = '&'
                dummy_node_id += 1

                self.add_edge_left(node, dummy_node_id)
                self.nodes[dummy_node_id]['smile'] = '&'
                self.nodes[dummy_node_id]['smarts'] = '&'
                dummy_node_id += 1
            else:
                raise ValueError('[CNXBinaryTree-make_full]: there are more than two children!')

        return
           
    def get_edge_by_start_node(self, node):
        edges_list = []

        edges = self.edges

        for eg in edges:
            if eg[0] == node:
                edges_list.append(eg)
        return edges_list 


    def BFS(self, source=None,  reverse=False, depth_limit=None, sort_neighbors=None):
        vlist = list(nx.bfs_tree(g, source = 0))

        return vlist

    def DFS(self, source=None, depth_limit=None):
        vlist = list(nx.dfs_tree(self, source = 0))
        return vlist
         

def test():
    data_dict = {
        0:[{'left':20, 'right':21}],
        20:[{'left':30, 'right':31}],
        21:[{'left':40, 'right':41}],
        30:[],
        31:[],
        40:[],
        41:[]
    }

    G = nx.DiGraph()

    # step 1: add edges
    for key in data_dict:
        print(key)
        for source in data_dict[key]:
            if 'left' in source:
                print('left [%d]' % (source['L']))  #left
                if source['left'] in data_dict:
                    G.add_edge(key, source['L'])
            if 'right' in source:
                print('right [%d]' % (source['R'])) #right
                if source['right'] in data_dict:
                    G.add_edge(key, source['R'])

    nx.draw_networkx(G)
    plt.show()
    print(G.edges())

    print(GTools.BFS(G))
    print(GTools.DFS(G))


if __name__ == '__main__':
    test()


