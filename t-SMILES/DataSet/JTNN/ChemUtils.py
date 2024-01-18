
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict

import rdkit.Chem as Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers

from MolUtils.RDKUtils.Utils import RDKUtils

class ChemUtils:
    MST_MAX_WEIGHT = 100 
    MAX_NCAND = 2000

    def set_atommap(mol, num=0):
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(num)

    def get_mol(smiles, kekuleSmiles = True):
        mol = Chem.MolFromSmiles(smiles)
        
        if kekuleSmiles:
            Chem.Kekulize(mol)

        if mol is None: 
            return None

        for a in mol.GetAtoms():
            if a.HasProp('molAtomMapNumber'):
                a.ClearProp('molAtomMapNumber')  

        return mol

    def get_smiles(mol):
        return Chem.MolToSmiles(mol, kekuleSmiles=True) 

    def decode_stereo(smiles2D):
        mol = Chem.MolFromSmiles(smiles2D)
        dec_isomers = list(EnumerateStereoisomers(mol))

        dec_isomers = [Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True)) for mol in dec_isomers]
        smiles3D = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in dec_isomers]

        chiralN = [atom.GetIdx() for atom in dec_isomers[0].GetAtoms() if int(atom.GetChiralTag()) > 0 and atom.GetSymbol() == "N"]
        if len(chiralN) > 0:
            for mol in dec_isomers:
                for idx in chiralN:
                    mol.GetAtomWithIdx(idx).SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
                smiles3D.append(Chem.MolToSmiles(mol, isomericSmiles=True))

        return smiles3D

    def sanitize(mol):
        try:
            smiles = ChemUtils.get_smiles(mol)
            mol = ChemUtils.get_mol(smiles)
        except Exception as e:
            return None
        return mol

    def copy_atom(atom):
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
        return new_atom

    def copy_edit_mol(mol):
        new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
        for atom in mol.GetAtoms():
            new_atom = ChemUtils.copy_atom(atom)
            new_mol.AddAtom(new_atom)
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            new_mol.AddBond(a1, a2, bt)
        return new_mol

    def get_clique_mol(mol, atoms, kekuleSmiles=True):              
        try:
            smiles = ''
            if kekuleSmiles:
                smiles  = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles = True)
                new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
                new_mol = ChemUtils.copy_edit_mol(new_mol).GetMol()
                new_mol = ChemUtils.sanitize(new_mol) 
                sml = Chem.MolToSmiles(new_mol, kekuleSmiles = True) 
            else:
                new_mol = ChemUtils.copy_edit_mol(mol).GetMol()
                smiles  = Chem.MolFragmentToSmiles(new_mol, atoms, kekuleSmiles = True)#
                new_mol = Chem.MolFromSmiles(smiles, sanitize=False)      
                sml = Chem.MolToSmiles(new_mol) 
        except Exception as e:
            print(f'[ChemUtils.get_clique_mol].exception[kekuleSmiles={kekuleSmiles}]:', e.args)

        return new_mol, sml

    def tree_decomp(mol):
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]], []

        cliques = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing():
                cliques.append([a1,a2])

        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        cliques.extend(ssr)

        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)
    
        for i in range(len(cliques)):
            if len(cliques[i]) <= 2: continue
            for atom in cliques[i]:
                for j in nei_list[atom]:
                    if i >= j or len(cliques[j]) <= 2: continue
                    inter = set(cliques[i]) & set(cliques[j])
                    if len(inter) > 2:
                        cliques[i].extend(cliques[j])
                        cliques[i] = list(set(cliques[i]))
                        cliques[j] = []
    
        cliques = [c for c in cliques if len(c) > 0]
        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)
    
        edges = defaultdict(int)
        for atom in range(n_atoms):
            if len(nei_list[atom]) <= 1: 
                continue
            cnei = nei_list[atom]
            bonds = [c for c in cnei if len(cliques[c]) == 2]
            rings = [c for c in cnei if len(cliques[c]) > 4]
            if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2): 
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1,c2)] = 1
            elif len(rings) > 2: 
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1,c2)] = ChemUtils.MST_MAX_WEIGHT - 1
            else:
                for i in range(len(cnei)):
                    for j in range(i + 1, len(cnei)):
                        c1,c2 = cnei[i],cnei[j]
                        inter = set(cliques[c1]) & set(cliques[c2])
                        if edges[(c1,c2)] < len(inter):
                            edges[(c1,c2)] = len(inter) 

        edges = [u + (ChemUtils.MST_MAX_WEIGHT-v,) for u,v in edges.items()]
        if len(edges) == 0:
            return cliques, edges

        row,col,data = zip(*edges)
        n_clique = len(cliques)

        clique_graph = csr_matrix( (data,(row,col)), shape=(n_clique,n_clique) )
        junc_tree = minimum_spanning_tree(clique_graph)

        row,col = junc_tree.nonzero()
        edges = [(row[i],col[i]) for i in range(len(row))]

        return (cliques, edges)

    def atom_equal(a1, a2):
        return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()

    def ring_bond_equal(b1, b2, reverse=False):
        b1 = (b1.GetBeginAtom(), b1.GetEndAtom())
        if reverse:
            b2 = (b2.GetEndAtom(), b2.GetBeginAtom())
        else:
            b2 = (b2.GetBeginAtom(), b2.GetEndAtom())
        return ChemUtils.atom_equal(b1[0], b2[0]) and ChemUtils.atom_equal(b1[1], b2[1])

    def attach_mols(ctr_mol,        
                    neighbors,     
                    prev_nodes,    
                    nei_amap        
                    ):
        prev_nids = [node.nid for node in prev_nodes]
        for nei_node in prev_nodes + neighbors:
            nei_id,nei_mol = nei_node.nid,nei_node.mol
            amap = nei_amap[nei_id]                         

            for atom in nei_mol.GetAtoms():
                if atom.GetIdx() not in amap:
                    new_atom = ChemUtils.copy_atom(atom)
                    amap[atom.GetIdx()] = ctr_mol.AddAtom(new_atom)

            if nei_mol.GetNumBonds() == 0:
                nei_atom = nei_mol.GetAtomWithIdx(0)
                ctr_atom = ctr_mol.GetAtomWithIdx(amap[0])
                ctr_atom.SetAtomMapNum(nei_atom.GetAtomMapNum())
            else:
                for bond in nei_mol.GetBonds():
                    a1 = amap[bond.GetBeginAtom().GetIdx()]
                    a2 = amap[bond.GetEndAtom().GetIdx()]

                    if ctr_mol.GetBondBetweenAtoms(a1, a2) is None:
                        ctr_mol.AddBond(a1, a2, bond.GetBondType())
                    elif nei_id in prev_nids: 
                        ctr_mol.RemoveBond(a1, a2)
                        ctr_mol.AddBond(a1, a2, bond.GetBondType())

        return ctr_mol

    def local_attach(ctr_mol,       
                     neighbors,     
                     prev_nodes,    
                     amap_list     
                     ):
        nb_mols= [nb.mol for nb in neighbors]
        nb_mols.insert(0,ctr_mol)

        ctr_mol = ChemUtils.copy_edit_mol(ctr_mol)
        nei_amap = {nei.nid:{} for nei in prev_nodes + neighbors}  

        for nei_id, ctr_atom, nei_atom in amap_list:
            nei_amap[nei_id][nei_atom] = ctr_atom                  

        ctr_mol = ChemUtils.attach_mols(ctr_mol,    
                                        neighbors, 
                                        prev_nodes, 
                                        nei_amap        
                                        ) 
        return ctr_mol.GetMol()

    def enum_attach_mol(ctr_mol, nei_node):
        return 

    def enum_attach(ctr_mol,    
                    nei_node,   
                    amap,       
                    singletons  
                    ):
        nei_mol,nei_idx = nei_node.mol, nei_node.nid

        att_confs = []
        black_list = [atom_idx for nei_id, atom_idx,_ in amap if nei_id in singletons]

        ctr_atoms = [atom for atom in ctr_mol.GetAtoms() if atom.GetIdx() not in black_list]  

        ctr_bonds = [bond for bond in ctr_mol.GetBonds()]

        if nei_mol.GetNumBonds() == 0: 
            nei_atom = nei_mol.GetAtomWithIdx(0)
            used_list = [atom_idx for _, atom_idx,_ in amap]
            for atom in ctr_atoms:
                if ChemUtils.atom_equal(atom, nei_atom) and atom.GetIdx() not in used_list:
                    new_amap = amap + [(nei_idx, atom.GetIdx(), 0)]
                    att_confs.append( new_amap )
   
        elif nei_mol.GetNumBonds() == 1: 
            bond = nei_mol.GetBondWithIdx(0)
            bond_val = int(bond.GetBondTypeAsDouble())
            b1,b2 = bond.GetBeginAtom(), bond.GetEndAtom()

            for atom in ctr_atoms: 
                if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() < bond_val:
                    continue
                if ChemUtils.atom_equal(atom, b1):
                    new_amap = amap + [(nei_idx, atom.GetIdx(), b1.GetIdx())]
                    att_confs.append( new_amap )
                elif ChemUtils.atom_equal(atom, b2):
                    new_amap = amap + [(nei_idx, atom.GetIdx(), b2.GetIdx())]
                    att_confs.append( new_amap )
        else: 
            for a1 in ctr_atoms:
                for a2 in nei_mol.GetAtoms():
                    if ChemUtils.atom_equal(a1, a2):
                        if a1.GetAtomicNum() == 6 and a1.GetTotalNumHs() + a2.GetTotalNumHs() < 4:
                            continue
                        new_amap = amap + [(nei_idx, a1.GetIdx(), a2.GetIdx())]
                        att_confs.append( new_amap )

            if ctr_mol.GetNumBonds() > 1:
                for b1 in ctr_bonds:
                    for b2 in nei_mol.GetBonds():
                        if ChemUtils.ring_bond_equal(b1, b2):
                            new_amap = amap + [(nei_idx, b1.GetBeginAtom().GetIdx(), b2.GetBeginAtom().GetIdx()),
                                               (nei_idx, b1.GetEndAtom().GetIdx(), b2.GetEndAtom().GetIdx())]
                            att_confs.append( new_amap )

                        if ChemUtils.ring_bond_equal(b1, b2, reverse=True):
                            new_amap = amap + [(nei_idx, b1.GetBeginAtom().GetIdx(), b2.GetEndAtom().GetIdx()),
                                               (nei_idx, b1.GetEndAtom().GetIdx(), b2.GetBeginAtom().GetIdx())]
                            att_confs.append( new_amap )

        return att_confs

    def enum_assemble(node, neighbors, prev_nodes=[], prev_amap=[]):
        all_attach_confs = []
        singletons = [nei_node.nid for nei_node in neighbors + prev_nodes if nei_node.mol.GetNumAtoms() == 1]
           
        def search(cur_amap, depth):
            if len(all_attach_confs) > ChemUtils.MAX_NCAND:
                return

            if depth == len(neighbors):
                all_attach_confs.append(cur_amap)
                return

            nei_node = neighbors[depth]                    
            cand_amap = ChemUtils.enum_attach(node.mol,    
                                              nei_node,     
                                              cur_amap,     
                                              singletons    
                                              )  
            cand_smiles = set()
            candidates = []
            for amap in cand_amap:
                cand_mol = ChemUtils.local_attach(node.mol,            
                                                  neighbors[:depth+1],  
                                                  prev_nodes, 
                                                  amap                 
                                                  )

                cand_mol = ChemUtils.sanitize(cand_mol)         

                if cand_mol is None:
                    continue
                smiles = ChemUtils.get_smiles(cand_mol)     
                if smiles in cand_smiles:
                    continue

                cand_smiles.add(smiles)
                candidates.append(amap)
                if len(candidates) > 3: 
                    print('ChemUtils[enum_assemble] len(candidates) > 3')
                    break

            if len(candidates) == 0:
                return
            else:
                for new_amap in candidates:
                    search(new_amap, depth + 1)
            #end search

        search(prev_amap, 0)

        cand_smiles = set()
        candidates = []
        for amap in all_attach_confs:
            cand_mol = ChemUtils.local_attach(node.mol, 
                                              neighbors, 
                                              prev_nodes, 
                                              amap
                                              )
            sml = Chem.MolToSmiles(cand_mol)
            cand_mol = Chem.MolFromSmiles(sml)
            smiles = Chem.MolToSmiles(cand_mol)
            if smiles in cand_smiles:
                continue
            cand_smiles.add(smiles)
            Chem.Kekulize(cand_mol)

            candidates.append((smiles, cand_mol, amap))
        return candidates

    def dfs_assemble(cur_mol, global_amap, fa_amap, cur_node, fa_node):
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors                   

        cur_amap = [(fa_nid,a2,a1) for nid,a1,a2 in fa_amap if nid == cur_node.nid]
        cands = ChemUtils.enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0: 
            return 

        cand_smiles,cand_mols,cand_amap = zip(*cands)
        label_idx = cand_smiles.index(cur_node.label)
        label_amap = cand_amap[label_idx]

        for nei_id,ctr_atom,nei_atom in label_amap:
            if nei_id == fa_nid:
                continue
            global_amap[nei_id][nei_atom] = global_amap[cur_node.nid][ctr_atom]
    
        cur_mol = ChemUtils.attach_mols(cur_mol, children, [], global_amap) 
        for nei_node in children:
            if not nei_node.is_leaf:
                ChemUtils.dfs_assemble(cur_mol, global_amap, label_amap, nei_node, cur_node)

        return 

    def dfs(stack, x, fa):
        for y in x.neighbors:
            if y.idx == fa.idx:
                continue
            stack.append((x,y,1))
            dfs(stack, y, x)
            stack.append((y,x,0))

    def have_slots(fa_slots, ch_slots):
        if len(fa_slots) > 2 and len(ch_slots) > 2:
            return True
        matches = []
        for i,s1 in enumerate(fa_slots):
            a1,c1,h1 = s1
            for j,s2 in enumerate(ch_slots):
                a2,c2,h2 = s2
                if a1 == a2 and c1 == c2 and (a1 != "C" or h1 + h2 >= 4):
                    matches.append( (i,j) )

        if len(matches) == 0: return False

        fa_match,ch_match = zip(*matches)
        if len(set(fa_match)) == 1 and 1 < len(fa_slots) <= 2: 
            fa_slots.pop(fa_match[0])
        if len(set(ch_match)) == 1 and 1 < len(ch_slots) <= 2: 
            ch_slots.pop(ch_match[0])

        return True
    
    def can_assemble(node_x, node_y):
        neis = node_x.neighbors + [node_y]
        for i,nei in enumerate(neis):
            nei.nid = i

        neighbors = [nei for nei in neis if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in neis if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cands = ChemUtils.enum_assemble(node_x, neighbors)

        return len(cands) > 0

    def decode_moltree(moltree):
        dec_smiles = ''
        try:
            moltree.recover()

            cur_mol = ChemUtils.copy_edit_mol(moltree.nodes[0].mol)
            global_amap = [{}] + [{} for node in moltree.nodes]
            global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}
            ChemUtils.dfs_assemble(cur_mol, global_amap, [], moltree.nodes[0], None)

            cur_mol = cur_mol.GetMol()    
            cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))

            ChemUtils.set_atommap(cur_mol)
            dec_smiles = Chem.MolToSmiles(cur_mol)

            RDKUtils.show_mol_with_atommap(cur_mol, atommap= False)  

        except Exception as e:
            print(e.args)

        return dec_smiles

if __name__ == "__main__":
    print('ChemUtils')
