import numpy as np

import sys
sys.path.append('/')
sys.path.append('../../ExternalGraph/JTVAE/')

import rdkit
import rdkit.Chem as Chem
import copy

from DataSet.JTNN.ChemUtils import ChemUtils
from MolUtils.RDKUtils.Frag.RDKFragUtil import Fragment_Alg
from MolUtils.RDKUtils.Frag.RDKFragBrics import RDKFragBrics
from MolUtils.RDKUtils.Frag.RDKFragMMPA import RDKFragMMPA
from MolUtils.RDKUtils.Frag.RDKFragScaffold import RDKFragScaffold


class MolTreeUtils:
    def get_slots(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]

class Vocab(object):

    def __init__(self, smiles_list):
        self.vocab = smiles_list
        self.vmap = {x:i for i,x in enumerate(self.vocab)}
        self.slots = [MolTreeUtils.get_slots(smiles) for smiles in self.vocab]
        
    def get_index(self, smiles):
        try:
            idx = self.vmap[smiles]
        except :
            idx = -1
        return idx

    def get_smiles(self, idx):
        smls = self.vocab[idx]
        return smls

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])

    def size(self):
        return len(self.vocab)

class MolTreeNode(object):

    def __init__(self, smiles, clique=[]):
        self.smiles = smiles
        self.mol = ChemUtils.get_mol(self.smiles)
        self.n_atoms = self.mol.GetNumAtoms()

        self.clique = [x for x in clique] #copy  , [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]
        self.neighbors = []
        
    def add_neighbor(self, nei_node):
        self.neighbors.append(nei_node)

    def degree(self):
        return len(self.neighbors)

    def recover(self, original_mol):
        clique = []
        clique.extend(self.clique)
        if not self.is_leaf:
            for cidx in self.clique:
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(self.nid)

        for nei_node in self.neighbors:
            clique.extend(nei_node.clique)
            if nei_node.is_leaf: #Leaf node, no need to mark 
                continue

            for cidx in nei_node.clique:
                #allow singleton node override the atom mapping
                if cidx not in self.clique or len(nei_node.clique) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node.nid)

        #RDKUtils.show_mol_with_atommap(original_mol, atommap= False)   

        clique = list(set(clique))
        label_mol, label_sml = ChemUtils.get_clique_mol(original_mol, clique)
        self.label = Chem.MolToSmiles(Chem.MolFromSmiles(ChemUtils.get_smiles(label_mol)))
        self.label_mol = ChemUtils.get_mol(self.label)

        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)

        return self.label
    
    def assemble(self):
        neighbors = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cands = ChemUtils.enum_assemble(self, neighbors)
        if len(cands) > 0:
            self.cands, self.cand_mols, _ = zip(*cands)
            self.cands = list(self.cands)
            self.cand_mols = list(self.cand_mols)
        else:
            self.cands = []
            self.cand_mols = []

class MolTree(object):
    def __init__(self, smiles, 
                dec_alg = Fragment_Alg.JTVAE,
                #dec_alg = Fragment_Alg.BRICS
                #dec_alg = Fragment_Alg.BRICS_Base #no extra ations besides brics
                #dec_alg = Fragment_Alg.MMPA
                #dec_alg = Fragment_Alg.Scaffold,
                kekuleSmiles = True, #updated at 2023.7.28 for for the reason:some kekuleSmiles can not be convert to mol
                 ):
        self.org_smiles = smiles
        self.kekuleSmiles = kekuleSmiles
        try:
            self.mol = ChemUtils.get_mol(smiles)  #Chem.Kekulize(mol)
            if self.mol is None:
                return 
            
            #Stereo Generation
            smls = Chem.MolToSmiles(self.mol) #, rootedAtAtom = 0
            mol = Chem.MolFromSmiles(smls)
            self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
            self.smiles2D = Chem.MolToSmiles(mol)
            self.stereo_cands = ChemUtils.decode_stereo(self.smiles2D)

            cliques = []
            edges = []

            s_atoms = [0]
            sn_atoms = 0
            sn_bonds = 0
            sn_clp = 0
            s_clp = [0]

            sub_smiles = self.org_smiles.split('.')
            n_sub = len(sub_smiles)
            
            for k in range(n_sub):
                sml = sub_smiles[k]
                if sml == ' ':
                    continue
                #s_mol = Chem.MolFromSmiles(sml)
                s_mol = ChemUtils.get_mol(sml)

                if dec_alg == Fragment_Alg.JTVAE:
                    s_cliques, s_edges = ChemUtils.tree_decomp(s_mol)              
                elif dec_alg == Fragment_Alg.BRICS:  #
                    #cliques, edges = brics_decomp(self.mol)
                    s_cliques, s_edges = RDKFragBrics.decompose_extra(s_mol)
                    if len(s_edges) <= 1:
                        print('Mol could not be Bricsed:', smiles)
                        s_cliques, s_edges = ChemUtils.tree_decomp(s_mol)
                elif dec_alg == Fragment_Alg.BRICS_Base:
                    s_cliques, s_edges = RDKFragBrics.decompose_simple(s_mol)
                elif dec_alg == Fragment_Alg.MMPA:
                    s_cliques, s_edges = RDKFragMMPA.decompose(s_mol)
                elif dec_alg == Fragment_Alg.Scaffold:
                    s_cliques, s_edges = RDKFragScaffold.decompose(s_mol)
                else:
                    #mol as one cliques
                    n_atoms = s_mol.GetNumAtoms()
                    s_cliques = [list(range(n_atoms))]
                    s_edges = []

                if k > 0:   #--patch for reaction: dot bond------
                    for i in range(s_cliques.__len__()):
                        for j in range(s_cliques[i].__len__()):
                            s_cliques[i][j] += sn_atoms
                    for i in range(len(s_edges)):
                        se = list(s_edges[i])
                        s_edges[i] = (se[0] + sn_clp , se[1] + sn_clp)

                sn_atoms += s_mol.GetNumAtoms()
                sn_bonds = len(mol.GetBonds())
                s_atoms.append(sn_atoms)

                sn_clp += len(s_cliques)
                s_clp.append(sn_clp)

                cliques.extend(s_cliques)
                edges.extend(s_edges)

            #--patch for reaction: dot bond------
            if k > 0:
                for i in range(1,len(s_clp)-1):
                    pos = s_clp[i]
                    edges.append((pos-1, pos))
            #------------------------------------
 
            self.nodes = []
            root = 0
            for i,c in enumerate(cliques):
                cmol, csmiles = ChemUtils.get_clique_mol(mol= self.mol, atoms = c, kekuleSmiles = kekuleSmiles)
                #sml2 = ChemUtils.get_smiles(cmol, kekuleSmiles=True, rootedAtAtom = 0)
                node = MolTreeNode(csmiles, c)
                self.nodes.append(node)
                if min(c) == 0:
                    root = i

            for x,y in edges:
                self.nodes[x].add_neighbor(self.nodes[y])
                self.nodes[y].add_neighbor(self.nodes[x])
        
            if root > 0:
                self.nodes[0],self.nodes[root] = self.nodes[root],self.nodes[0]

            for i,node in enumerate(self.nodes):
                node.nid = i + 1  #why +1 ????
                #node.nid = i
                if len(node.neighbors) > 1: #Leaf node mol is not marked
                    ChemUtils.set_atommap(node.mol, node.nid)
                node.is_leaf = (len(node.neighbors) == 1)
        except Exception as e:
            print(e.args)
            self.mol = None
        return 

    def size(self):
        return len(self.nodes)

    def recover(self):
        for node in self.nodes:
            node.recover(self.mol)

    def assemble(self):
        for node in self.nodes:
            node.assemble()


def preprocess_smiles(smlfile, dec_alg = Fragment_Alg.Scaffold): 
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv(smlfile, squeeze=True, delimiter=',',header = None) 
    #df.dropna(how="any")
    #smiles_list = [s for s in df.values.astype(str) if s != 'nan']
    smiles_list = list(df.values)

    vocab_list = set()
    for i, sml in tqdm(enumerate(smiles_list), desc = 'parsing smiles to create voc...', total = len(smiles_list)):
        moltree = MolTree(sml, dec_alg = dec_alg)
        if moltree.mol is not None:
            for c in moltree.nodes:
                vocab_list.add(c.smiles)
        else:
            print('There are something wrong with smiles:', sml)

    vocab = sorted(vocab_list)


    vocfile = smlfile+'.'+ dec_alg +'_token.voc'

    df = pd.DataFrame(vocab)
    df.to_csv(vocfile, index = False, header=False, na_rep="NULL")
    return 


if __name__ == "__main__":
    import sys
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    smlfile = '../RawData/AID1706/active.smi'       

    preprocess_smiles(smlfile,
                    dec_alg = Fragment_Alg.JTVAE,
                    #dec_alg = Fragment_Alg.BRICS
                    #dec_alg = Fragment_Alg.BRICS_Base #no extra ations besides brics
                    #dec_alg = Fragment_Alg.MMPA
                    #dec_alg = Fragment_Alg.Scaffold,
                    )
