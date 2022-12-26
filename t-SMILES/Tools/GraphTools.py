import networkx as nx
import matplotlib.pyplot as plt

class GTools:
    def BFS(g, source = 0):# Breadth_firct Search
        #g = nx.fast_gnp_random_graph(10, 0.2)
        vlist = list(nx.bfs_tree(g, source = source))
        return vlist

    def DFS(g, source = 0, depth_limit = 3):#depth-First Search
        #g = nx.fast_gnp_random_graph(10, 0.2)
        vlist = list(nx.dfs_tree(g, source = source))#, depth_limit = depth_limit
        #vlist = [list(nx.dfs_tree(g, node)) for node in [1,4,5]]
        return vlist
