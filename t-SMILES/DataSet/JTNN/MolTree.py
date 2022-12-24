import sys
sys.path.append('/')
sys.path.append('../../ExternalGraph/JTVAE/')

import rdkit
import rdkit.Chem as Chem
import copy

from DataSet.JTNN.ChemUtils import ChemUtils
from MolUtils.RDKUtils.Utils import RDKUtils

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
        label_mol = ChemUtils.get_clique_mol(original_mol, clique)
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
                 dec_alg = 'JTVAE'
                 #dec_alg = 'BRICS'
                 ):
        self.smiles = smiles
        try:
            self.mol = ChemUtils.get_mol(smiles)  #Chem.Kekulize(mol)
            
            #Stereo Generation
            mol = Chem.MolFromSmiles(smiles)
            self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
            self.smiles2D = Chem.MolToSmiles(mol)
            self.stereo_cands = ChemUtils.decode_stereo(self.smiles2D)

            if dec_alg == 'JTVAE':
                cliques, edges = ChemUtils.tree_decomp(self.mol)              
            else:  #alg == 'BRICS':
                cliques, edges = ChemUtils.brics_decomp_extra(self.mol)
                if len(edges) <= 1:
                    print('Mol could not be Bricsed! ')
                    cliques, edges = ChemUtils.tree_decomp(self.mol)
            
            self.nodes = []
            root = 0
            for i,c in enumerate(cliques):
                cmol = ChemUtils.get_clique_mol(self.mol, c)
                node = MolTreeNode(ChemUtils.get_smiles(cmol), c)
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


def preprocess_smiles(smlfile, dec_alg = 'BRICS'): 
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

def preprocess_sml(dec_alg = 'BRICS'):
    #sml = 'C\C(=C1\SC(=O)N(C1=O)c1ccc(Cl)cc1)c1ccc(Br)cc1' #could not brics
    #sml = 'FC(F)(F)CCCCc1ccc2Cn3cncc3CCN3CCN(C(=O)C3)c3cccc4ccc(Oc1c2)cc34' #could not brics
    #sml = 'O=[N+]([O-])c1c(Nc2cccc3ncccc23)ncnc1N1CCN(c2cccc(Cl)c2)CC1' #could not brics
    #sml = 'CN(C)C=C1C(=O)N(C(c2cccnc2)S1(=O)=O)c1ccc(F)cc1F' #could not brics
    #sml = 'CC(C)N(CC(C)(C)O)C(=O)NC(C1CC1)c1cccc(c1)C(F)(F)F' #could be bricsed
    sml='C1(F)[CH]C=CC=C1OCC(N)O'#could be bricsed

    moltree = MolTree(sml, dec_alg = dec_alg)
    if moltree.mol is not None:
        for c in moltree.nodes:
            vocab_list.add(c.smiles)

    return 

if __name__ == "__main__":
    import sys
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)


    #cset = set()
    #for i,line in enumerate(sys.stdin):
    #    smiles = line.split()[0]
    #    mol = MolTree(smiles)
    #    for c in mol.nodes:
    #        cset.add(c.smiles)
    #for x in cset:
    #    print( x)


    #smlfile = '../RawData/special.smi'

    smlfile = '../RawData/JTVAE/data/zinc/all.txt'
    #smlfile = '../RawData/JTVAE/data/zinc/all.txt.bfs[64]_org.smi'    

    #smlfile = '../RawData/ChEMBL/chembl_21_1576904.csv_Canonical.smi'
    #smlfile = '../RawData/ChEMBL/long_ring_1.smi'
    #smlfile = '../RawData/ChEMBL/chembl_21_1576904.csv_Canonical.smi.scaffold_scaf.smi'
    #smlfile = '../RawData/ChEMBL/chembl_120_Canonical.smi.bfs[228]_org.smi'
    #smlfile = '../RawData/ChEMBL/Brics/test.smi'

    #smlfile = '../RawData/QM/QM9/QM9.smi'       
    #smlfile = '../RawData/Antiviral/coronavirus_data/conversions/AID1706.smi'       

    preprocess_smiles(smlfile,
                    #dec_alg = 'JTVAE'
                    dec_alg = 'BRICS'
                    )

    #preprocess_sml( 
    #                dec_alg = 'JTVAE'
    #                #dec_alg = 'BRICS'
    #                )
