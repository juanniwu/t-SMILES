import networkx as nx


class GTools:
    def BFS(g, source = 0):
        vlist = list(nx.bfs_tree(g, source = source))
        return vlist

    def DFS(g, source = 0, depth_limit = 3):
        vlist = list(nx.dfs_tree(g, source = source))
        return vlist
