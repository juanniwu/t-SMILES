import numpy as np
import copy

import rdkit.Chem as Chem

from DataSet.JTNN.ChemUtils import ChemUtils
from MolUtils.RDKUtils.Utils import RDKUtils


from MolUtils.RDKUtils.Frag.RDKFragUtil import Fragment_Alg
from MolUtils.RDKUtils.Frag.RDKFragBrics import RDKFragBrics
from MolUtils.RDKUtils.Frag.RDKFragBricsDummy import RDKFragBricsDummy
from MolUtils.RDKUtils.Frag.RDKFragMMPA import RDKFragMMPA
from MolUtils.RDKUtils.Frag.RDKFragScaffold import RDKFragScaffold
from MolUtils.RDKUtils.Frag.RDKFragRBrics import RDKFragRBrics


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

    def __init__(self, smiles, clique=[], smarts = None, kekuleSmiles = True, frg_random = False):
        self.smiles = smiles
        self.mol = ChemUtils.get_mol(self.smiles, kekuleSmiles)
        self.n_atoms = self.mol.GetNumAtoms()

        self.clique = [x for x in clique] 
        self.neighbors = []
        self.smarts = smarts

        self.kekuleSmiles = kekuleSmiles

        return 
        
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
            if nei_node.is_leaf: 
                continue

            for cidx in nei_node.clique:
                if cidx not in self.clique or len(nei_node.clique) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node.nid)

        clique = list(set(clique))
        label_mol, label_sml = ChemUtils.get_clique_mol(original_mol, clique)
        self.label = Chem.MolToSmiles(Chem.MolFromSmiles(ChemUtils.get_smiles(label_mol)))
        self.label_mol = ChemUtils.get_mol(self.label, self.kekuleSmiles)

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
                kekuleSmiles = True, #updated at 2023.7.28 for for the reason:some kekuleSmiles can not be convert to mol
                frg_random   = False,
                ):
        self.org_smiles = smiles
        self.kekuleSmiles = kekuleSmiles
        self.frg_random = frg_random
        try:
            show = False

            self.org_mol = ChemUtils.get_mol(smiles, kekuleSmiles)  
            if self.org_mol is None:
                return 

            if show:
                RDKUtils.show_mol_with_atommap(self.mol, atommap = True)

            
            self.smiles = Chem.MolToSmiles(self.org_mol, kekuleSmiles = kekuleSmiles)
            self.smarts = Chem.MolToSmarts(self.org_mol, isomericSmiles=False)
            self.mol = ChemUtils.get_mol(self.smiles, kekuleSmiles) 

            self.atom_env = None

            #Stereo Generation
            smls = Chem.MolToSmiles(self.mol) 
            mol = Chem.MolFromSmiles(smls)
            self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
            self.smiles2D = Chem.MolToSmiles(mol)
            self.stereo_cands = ChemUtils.decode_stereo(self.smiles2D)

            self.atoms = self.mol.GetAtoms()     
            self.n_atoms =  self.mol.GetNumAtoms()
            self.atom_idx = [atom.GetIdx() for atom in self.atoms]
            self.atmomic_nums = [atom.GetAtomicNum() for atom in self.atoms] 
            self.atom_symbols = [atom.GetSymbol() for atom in self.atoms]   
            self.atmomic_nums_dict = sorted(set([atom.GetAtomicNum() for atom in self.atoms] + [0]))
            self.atom_n_type = len(self.atmomic_nums_dict)

            self.bonds = self.mol.GetBonds()
            self.n_bonds = len(self.bonds)
            self.bonds_s = [bond.GetBondType() for bond in self.bonds]
            self.bond_labels = [Chem.rdchem.BondType.ZERO] + \
                    list(sorted(set(bond.GetBondType() for bond in self.bonds)))
            self.bond_n_types = len(self.bond_labels)
               
            begin = [b.GetBeginAtomIdx() for b in self.bonds]
            end = [b.GetEndAtomIdx() for b in self.bonds]
            self.bond_pair_list = np.column_stack((np.asarray(begin), np.asarray(end)))            

            cliques = []
            edges = []

            s_atoms = [0]
            sn_atoms = 0
            sn_bonds = 0
            sn_clp = 0
            s_clp = [0]

            self.atom_env = []
            self.cut_bonds = []

            sub_smiles = self.smiles.split('.')   
            n_sub = len(sub_smiles)
            
            for k in range(n_sub):
                sml = sub_smiles[k]
                if sml == ' ':
                    continue
                s_mol = ChemUtils.get_mol(sml, kekuleSmiles)
                n_atoms = s_mol.GetNumAtoms()

                if dec_alg == Fragment_Alg.Vanilla:
                    s_cliques = [list(range(n_atoms))]
                    s_edges = []
                    motif_str = [self.smiles]
                elif dec_alg == Fragment_Alg.JTVAE:
                    s_cliques, s_edges = ChemUtils.tree_decomp(s_mol)              
                elif dec_alg == Fragment_Alg.BRICS:  
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

                elif dec_alg == Fragment_Alg.BRICS_DY:
                    s_cliques, s_edges, motif_str, dummy_atom, motif_smarts = RDKFragBricsDummy.decompose_dummy(s_mol)
                elif dec_alg == Fragment_Alg.MMPA_DY:
                    s_cliques, s_edges, motif_str, dummy_atom, motif_smarts = RDKFragMMPA.decompose_dummy(s_mol)
                elif dec_alg == Fragment_Alg.Scaffold_DY:
                    s_cliques, s_edges, motif_str, dummy_atom, motif_smarts = RDKFragScaffold.decompose_dummy(s_mol)
                elif dec_alg == Fragment_Alg.RBrics_DY:
                    s_cliques, s_edges, motif_str, dummy_atom, motif_smarts = RDKFragRBrics.decompose_dummy(s_mol)
                else:
                    s_cliques = [list(range(n_atoms))]
                    s_edges = []

                if k > 0:   
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

            #--patch for reaction: 
            if k > 0:
                for i in range(1,len(s_clp)-1):
                    pos = s_clp[i]
                    edges.append((pos-1, pos))

            #--------

            self.edges = edges
 
            self.nodes = []
            root = 0
            for i, c in enumerate(cliques):
                if dec_alg == Fragment_Alg.Vanilla:
                    csmiles = motif_str[i]
                    csmarts = csmiles
                elif dec_alg == Fragment_Alg.BRICS_DY or dec_alg == Fragment_Alg.MMPA_DY or \
                    dec_alg == Fragment_Alg.Scaffold_DY or dec_alg == Fragment_Alg.RBrics_DY:
                    csmiles = motif_str[i]
                    csmarts = motif_smarts[i]
                else:
                    cmol, csmiles = ChemUtils.get_clique_mol(mol= self.mol, atoms = c, kekuleSmiles = True)
                    csmarts = csmiles
                
                node = MolTreeNode(csmiles, c, csmarts, kekuleSmiles)
                self.nodes.append(node)
                if min(c) == 0:
                    root = i

            for x, y in edges:
                self.nodes[x].add_neighbor(self.nodes[y])
                self.nodes[y].add_neighbor(self.nodes[x])
        
            if root > 0:
                self.nodes[0], self.nodes[root] = self.nodes[root], self.nodes[0]

            for i,node in enumerate(self.nodes):
                node.nid = i + 1  
                if len(node.neighbors) > 1: 
                    ChemUtils.set_atommap(node.mol, node.nid)

                node.is_leaf = (len(node.neighbors) == 1)

        except Exception as e:
            print('[MolTree.init].Exception:',e.args)
            print('[MolTree.init].Exception-smiles:',smiles)
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

#-----------------------------------------------
