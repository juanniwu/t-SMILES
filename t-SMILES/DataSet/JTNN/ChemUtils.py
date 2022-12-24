import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import BRICS

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict

from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

from MolUtils.RDKUtils.Utils import RDKUtils

class ChemUtils:
    MST_MAX_WEIGHT = 100 
    MAX_NCAND = 2000

    def set_atommap(mol, num=0):
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(num)

    def get_mol(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: 
            return None
        Chem.Kekulize(mol)
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

    def get_clique_mol(mol, atoms):
        smiles  = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
        new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
        new_mol = ChemUtils.copy_edit_mol(new_mol).GetMol()
        new_mol = ChemUtils.sanitize(new_mol) #We assume this is not None
        return new_mol

    def tree_decomp(mol):
        #RDKUtils.show_mol_with_atommap(mol, atommap= True)  

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
    
        #Merge Rings with intersection > 2 atoms
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
    
        #Build edges and add singleton cliques
        edges = defaultdict(int)
        for atom in range(n_atoms):
            if len(nei_list[atom]) <= 1: 
                continue
            cnei = nei_list[atom]
            bonds = [c for c in cnei if len(cliques[c]) == 2]
            rings = [c for c in cnei if len(cliques[c]) > 4]
            if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2): #In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1,c2)] = 1
            elif len(rings) > 2: #Multiple (n>2) complex rings
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
                            edges[(c1,c2)] = len(inter) #cnei[i] < cnei[j] by construction

        edges = [u + (ChemUtils.MST_MAX_WEIGHT-v,) for u,v in edges.items()]
        if len(edges) == 0:
            return cliques, edges

        #Compute Maximum Spanning Tree
        row,col,data = zip(*edges)
        n_clique = len(cliques)
        clique_graph = csr_matrix( (data,(row,col)), shape=(n_clique,n_clique) )
        junc_tree = minimum_spanning_tree(clique_graph)
        row,col = junc_tree.nonzero()
        edges = [(row[i],col[i]) for i in range(len(row))]
        return (cliques, edges)

    def atom_equal(a1, a2):
        return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()

    #Bond type not considered because all aromatic (so SINGLE matches DOUBLE)
    def ring_bond_equal(b1, b2, reverse=False):
        b1 = (b1.GetBeginAtom(), b1.GetEndAtom())
        if reverse:
            b2 = (b2.GetEndAtom(), b2.GetBeginAtom())
        else:
            b2 = (b2.GetBeginAtom(), b2.GetEndAtom())
        return ChemUtils.atom_equal(b1[0], b2[0]) and ChemUtils.atom_equal(b1[1], b2[1])

    def attach_mols(ctr_mol,        #rdchem.RWMol 'C[NH3+]'
                    neighbors,      #MolTreeNode, [0]:'[NH4+]'
                    prev_nodes,     # []
                    nei_amap        #{2: {0: 1}} #nei_id, ctr_atom, nei_atom in amap_list
                    ):
        #[attach_mols---in:ctr_mol]  C[NH3+]
        #[attach_mols---in:neighbors-0] CN
        #[attach_mols---in:neighbors-1] C=O
        #[attach_mols---in:neighbors-2] CC
        #[attach_mols:ctr_mol---output] CC(N)=O

        #print('[attach_mols---in:ctr_mol]',Chem.MolToSmiles(ctr_mol))#jw           
        #for i in range(len(neighbors)):  
        #    molb = neighbors[i]         #for debuging
        #    print(f'[attach_mols---in:neighbors-{i}]',Chem.MolToSmiles(molb.mol))#jw

        prev_nids = [node.nid for node in prev_nodes]
        for nei_node in prev_nodes + neighbors:
            nei_id,nei_mol = nei_node.nid,nei_node.mol
            amap = nei_amap[nei_id]                         #{0: 1} ctr_atom, nei_atom

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
                    elif nei_id in prev_nids: #father node overrides
                        ctr_mol.RemoveBond(a1, a2)
                        ctr_mol.AddBond(a1, a2, bond.GetBondType())

        #print('[attach_mols:ctr_mol---output]',Chem.MolToSmiles(ctr_mol))#jw
        return ctr_mol

    def local_attach(ctr_mol,       #rdchem.Mol  :'C[NH3+]'
                     neighbors,     #MolTreeNode : '[NH4+]'
                     prev_nodes,    #[]
                     amap_list      #[(2, 1, 0)]
                     ):
        nb_mols= [nb.mol for nb in neighbors]
        nb_mols.insert(0,ctr_mol)
        #RDKUtils.show_mol_with_atommap(nb_mols, atommap= False)  

        #print('local_attach.ctr_mol', Chem.MolToSmiles(ctr_mol))
        ctr_mol = ChemUtils.copy_edit_mol(ctr_mol)
        nei_amap = {nei.nid:{} for nei in prev_nodes + neighbors}   #{2: {}}

        for nei_id, ctr_atom, nei_atom in amap_list:
            nei_amap[nei_id][nei_atom] = ctr_atom                   #{2: {0: 1}}

        ctr_mol = ChemUtils.attach_mols(ctr_mol,    
                                        neighbors, 
                                        prev_nodes, 
                                        nei_amap        #nei_id, ctr_atom, nei_atom in amap_list
                                        ) #'C[NH3+:2]'

        #RDKUtils.show_mol_with_atommap(ctr_mol, atommap= False)  
        return ctr_mol.GetMol()

    def enum_attach_mol(ctr_mol, nei_node):
        return 

    #This version records idx mapping between ctr_mol and nei_mol
    def enum_attach(ctr_mol,    #rdchem.Mol 
                    nei_node,   #MolTreeNode, '[NH4+]'
                    amap,       #[]
                    singletons  #[2]
                    ):
        #print('enum_attach.ctr_mol = ', Chem.MolToSmiles(ctr_mol)) #'C[NH3+]'
        #print('enum_attach.nei_node = ', nei_node.smiles) #'C[NH3+]'

        nei_mol,nei_idx = nei_node.mol, nei_node.nid
        #print('[enum_attach:nei_mol]',Chem.MolToSmiles(nei_mol))#jw

        att_confs = []
        black_list = [atom_idx for nei_id, atom_idx,_ in amap if nei_id in singletons] #[]

        ctr_atoms = [atom for atom in ctr_mol.GetAtoms() if atom.GetIdx() not in black_list]  #atom('C'), atom('C')
        #ctr_atoms[0]:rdchem.Atom: .GetSymbol() = 'C'
        #ctr_atoms[0]:rdchem.Atom: .GetSymbol() = 'N'

        ctr_bonds = [bond for bond in ctr_mol.GetBonds()]

        if nei_mol.GetNumBonds() == 0: #neighbor singleton
            nei_atom = nei_mol.GetAtomWithIdx(0)
            used_list = [atom_idx for _, atom_idx,_ in amap]
            for atom in ctr_atoms:
                if ChemUtils.atom_equal(atom, nei_atom) and atom.GetIdx() not in used_list:
                    new_amap = amap + [(nei_idx, atom.GetIdx(), 0)]
                    att_confs.append( new_amap )
   
        elif nei_mol.GetNumBonds() == 1: #neighbor is a bond
            bond = nei_mol.GetBondWithIdx(0)
            bond_val = int(bond.GetBondTypeAsDouble())
            b1,b2 = bond.GetBeginAtom(), bond.GetEndAtom()
            #b1.GetSymbol() = 'C', b2.GetSymbol() = 'C'
            for atom in ctr_atoms: 
                #Optimize if atom is carbon (other atoms may change valence)
                if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() < bond_val:
                    continue
                if ChemUtils.atom_equal(atom, b1):
                    new_amap = amap + [(nei_idx, atom.GetIdx(), b1.GetIdx())]
                    att_confs.append( new_amap )
                elif ChemUtils.atom_equal(atom, b2):
                    new_amap = amap + [(nei_idx, atom.GetIdx(), b2.GetIdx())]
                    att_confs.append( new_amap )
        else: 
            #intersection is an atom
            for a1 in ctr_atoms:
                for a2 in nei_mol.GetAtoms():
                    if ChemUtils.atom_equal(a1, a2):
                        #Optimize if atom is carbon (other atoms may change valence)
                        if a1.GetAtomicNum() == 6 and a1.GetTotalNumHs() + a2.GetTotalNumHs() < 4:
                            continue
                        new_amap = amap + [(nei_idx, a1.GetIdx(), a2.GetIdx())]
                        att_confs.append( new_amap )

            #intersection is an bond
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

        #print('enum_attach.att_confs = ', att_confs) #'C[NH3+]'
        return att_confs

    #Try rings first: Speed-Up 
    def enum_assemble(node, neighbors, prev_nodes=[], prev_amap=[]):
        #print('---[enum_assemble:node]--------------------------------')#jw     'C[NH3+]'
        #print('enum_assemble:node.smiles',node.smiles)   
        #print('enum_assemble:neighbors',neighbors)                      #        '[NH4+]'
        #print('enum_assemble:prev_nodes',prev_nodes)  
        #print('enum_assemble:prev_amap',prev_amap) 
        #print('-------------------end of enum_assemble.input------------------------')             

        #print('[enum_assemble:node]',Chem.MolToSmiles(node.mol))#jw for testing
        #for i in range(len(neighbors)):
        #    molb = neighbors[i]
        #    print(f'[enum_assemble:neighbors-{i}]',Chem.MolToSmiles(molb.mol))#jw

        all_attach_confs = []
        singletons = [nei_node.nid for nei_node in neighbors + prev_nodes if nei_node.mol.GetNumAtoms() == 1]
           
        def search(cur_amap, depth):
            #node, nei_node, cur_amap, singletons, prev_nodes
            #neighbors
            #all_attach_confs
            #cur_amap(nei_id, ctr_atom, nei_atom)

            if len(all_attach_confs) > ChemUtils.MAX_NCAND: #2000
                return

            if depth == len(neighbors):
                all_attach_confs.append(cur_amap)
                return

            nei_node = neighbors[depth]                     #nei_node.smiles = '[NH4+]'
            cand_amap = ChemUtils.enum_attach(node.mol,     #'C[NH3+]'
                                              nei_node,     #MolTreeNode, 
                                              cur_amap,     #[]
                                              singletons    #[0]=2
                                              )  #return something like [(2, 1, 0)]
            cand_smiles = set()
            candidates = []
            for amap in cand_amap:
                #amap = [(2, 1, 0)]
                cand_mol = ChemUtils.local_attach(node.mol,             #'C[NH3+]'
                                                  neighbors[:depth+1],  #'[NH4+]'
                                                  prev_nodes, 
                                                  amap                  #nei_id, ctr_atom, nei_atom in amap_list
                                                  )
                #print('[enum_assemble:cand_mol]',Chem.MolToSmiles(cand_mol))#jw

                cand_mol = ChemUtils.sanitize(cand_mol)         #'C[NH3+:2]'
                #print('[enum_assemble:cand_mol]',Chem.MolToSmiles(cand_mol))#jw

                if cand_mol is None:
                    continue
                smiles = ChemUtils.get_smiles(cand_mol)     #'C[NH3+:2]'
                if smiles in cand_smiles:
                    continue

                cand_smiles.add(smiles)
                candidates.append(amap)

            if len(candidates) == 0:
                return

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
            #print('enum_assemble:search.sml',sml)   
            
            #RDKUtils.show_mol_with_atommap(cand_mol, atommap= False)  

            cand_mol = Chem.MolFromSmiles(sml)
            smiles = Chem.MolToSmiles(cand_mol)
            if smiles in cand_smiles:
                continue
            cand_smiles.add(smiles)
            Chem.Kekulize(cand_mol)

            candidates.append((smiles, cand_mol, amap))
            #smiles: 'C=O' or 'COC'
            #cand_mol: <rdkit.Chem.rdchem.Mol object at 0x00000251923B89E0>
            #amap: [(0, 0, 0)] or [(0, 0, 0), (1, 1, 1)]

        #print('---[enum_assemble:candidates].return:', candidates) 
        return candidates

    #Only used for debugging purpose
    def dfs_assemble(cur_mol, global_amap, fa_amap, cur_node, fa_node):
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors
                   
        #nb_mols= [nb.mol for nb in neighbors]
        #nb_mols.insert(0, cur_node.mol)
        #RDKUtils.show_mol_with_atommap(nb_mols, atommap= False)  


        cur_amap = [(fa_nid,a2,a1) for nid,a1,a2 in fa_amap if nid == cur_node.nid]
        cands = ChemUtils.enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0: 
            return 

        #cand_smiles,cand_amap = zip(*cands)
        cand_smiles,cand_mols,cand_amap = zip(*cands)
        #
        #
        #cand_amap:[(13, 1, 0)] 13 means frag.nid

        label_idx = cand_smiles.index(cur_node.label)
        label_amap = cand_amap[label_idx]

        for nei_id,ctr_atom,nei_atom in label_amap:
            if nei_id == fa_nid:
                continue
            global_amap[nei_id][nei_atom] = global_amap[cur_node.nid][ctr_atom]
    
        cur_mol = ChemUtils.attach_mols(cur_mol, children, [], global_amap) #father is already attached
        for nei_node in children:
            if not nei_node.is_leaf:
                ChemUtils.dfs_assemble(cur_mol, global_amap, label_amap, nei_node, cur_node)

        return 

    #moved from JTNNDecoder
    def dfs(stack, x, fa):
        for y in x.neighbors:
            if y.idx == fa.idx:
                continue
            stack.append((x,y,1))
            ChemUtils.dfs(stack, y, x)
            stack.append((y,x,0))

    #moved from JTNNDecoder
    def have_slots(fa_slots, ch_slots):
        #print('have_slots:fa_slots', fa_slots)
        #print('have_slots:ch_slots', ch_slots)

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
        if len(set(fa_match)) == 1 and 1 < len(fa_slots) <= 2: #never remove atom from ring
            fa_slots.pop(fa_match[0])
        if len(set(ch_match)) == 1 and 1 < len(ch_slots) <= 2: #never remove atom from ring
            ch_slots.pop(ch_match[0])

        return True
    
    #moved from JTNNDecoder
    def can_assemble(node_x, node_y):
        #print('ChemUtils.can_assemble:node_x', node_x.smiles)
        #print('ChemUtils.can_assemble:node_y', node_y.smiles)

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

            #RDKUtils.show_mol_with_atommap(cur_mol, atommap= False)  

            global_amap = [{}] + [{} for node in moltree.nodes]
            global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}

            ChemUtils.dfs_assemble(cur_mol, global_amap, [], moltree.nodes[0], None)

            #RDKUtils.show_mol_with_atommap(cur_mol, atommap= False)  

            cur_mol = cur_mol.GetMol()
    
            cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))

            ChemUtils.set_atommap(cur_mol)
            dec_smiles = Chem.MolToSmiles(cur_mol)

            #print('decode_test_sml.dec_smiles = ',dec_smiles)
            #RDKUtils.show_mol_with_atommap(cur_mol, atommap= False)

        except Exception as e:
            print(e.args)

        return dec_smiles


    #-----------------------------------------

    def find_bond(mol_bonds, startid, endid):
        for b in mol_bonds:
            if b.GetBeginAtomIdx() == startid and b.GetEndAtomIdx() == endid:
                return b
        return None

    def bond_in_ring(mol_bonds, startid, endid):
        b = ChemUtils.find_bond(mol_bonds, startid, endid)
        if b is not None and b.IsInRing():
            return True
        return False

    def bond_is_bridge(bond, mol):
        s = bond.GetBeginAtomIdx()
        e = bond.GetEndAtomIdx()

        if not bond.IsInRing() and mol.GetAtomWithIdx(s).IsInRing() and mol.GetAtomWithIdx(e).IsInRing():
            return True
        else:
            return False

    def brics_decomp_extra(mol, break_long_link=True, break_r_bridge=True):
        # RDKUtils.show_mol_with_atommap(mol, atommap= True)

        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]], []

        cliques = []
        breaks = []

        atom_cliques = {}
        for i in range(n_atoms):
            atom_cliques[i] = set()

        try:
            mol_bonds = mol.GetBonds()
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom().GetIdx()
                a2 = bond.GetEndAtom().GetIdx()
                cliques.append([a1, a2])

            single_cliq = []

            brics_bonds = list(BRICS.FindBRICSBonds(mol))
            if len(brics_bonds) == 0:
                return [list(range(n_atoms))], []
            else:
                for bond in brics_bonds:
                    bond = bond[0]
                    if [bond[0], bond[1]] in cliques:
                        cliques.remove([bond[0], bond[1]])
                    else:
                        cliques.remove([bond[1], bond[0]])

                    atom = mol.GetAtomWithIdx(bond[0])
                    if not atom.IsInRing():
                        if len(atom.GetNeighbors()) > 2:
                            if [atom.GetIdx()] not in single_cliq:
                                single_cliq.append([atom.GetIdx()])
                        else:
                            cliques.append([bond[0]])
                    atom = mol.GetAtomWithIdx(bond[1])
                    if not atom.IsInRing():
                        if len(atom.GetNeighbors()) > 2:
                            if [atom.GetIdx()] not in single_cliq:
                                single_cliq.append([atom.GetIdx()])
                        else:
                            cliques.append([bond[1]])

            # break bonds between rings and non-ring atoms,  non-ring and non-ring
            a_not_in_ring = []
            for c in cliques:
                if len(c) > 1:
                    if mol.GetAtomWithIdx(c[0]).IsInRing() and not mol.GetAtomWithIdx(c[1]).IsInRing():
                        breaks.append(c)
                        a_not_in_ring.append(c[1])
                    if mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(c[0]).IsInRing():
                        breaks.append(c)
                        a_not_in_ring.append(c[0])
                    if break_long_link:
                        if not mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(
                                c[0]).IsInRing():  # non-ring and non-ring
                            breaks.append(c)
                            a_not_in_ring.append(c[0])
                            a_not_in_ring.append(c[1])
                    if break_r_bridge:
                        if mol.GetAtomWithIdx(c[0]).IsInRing() and mol.GetAtomWithIdx(
                                c[1]).IsInRing():  # ring-ring bridge
                            if not ChemUtils.bond_in_ring(mol_bonds, c[0], c[1]):
                                breaks.append(c)
                                # a_not_in_ring.append(c[0])
                                # a_not_in_ring.append(c[1])

            for b in breaks:
                if b in cliques:
                    cliques.remove(b)
            for a in a_not_in_ring:
                atom = mol.GetAtomWithIdx(a)
                if len(atom.GetNeighbors()) > 2:
                    if [atom.GetIdx()] not in single_cliq:
                        single_cliq.append([atom.GetIdx()])
                else:
                    if [a] not in cliques:
                        cliques.append([a])

            # select atoms at intersections as motif
            for atom in mol.GetAtoms():
                if len(atom.GetNeighbors()) > 2 and not atom.IsInRing():
                    aid = atom.GetIdx()
                    # cliques.append([atom.GetIdx()])
                    if [aid] not in single_cliq:
                        single_cliq.append([aid])

                    for nei in atom.GetNeighbors():
                        nid = nei.GetIdx()

                        if [nid, aid] in cliques:
                            cliques.remove([nid, aid])
                            breaks.append([nid, aid])
                        elif [aid, nid] in cliques:
                            cliques.remove([aid, nid])
                            breaks.append([aid, nid])

                        if len(nei.GetNeighbors()) > 2 and not nei.IsInRing():
                            if [nid] not in single_cliq:
                                single_cliq.append([nid])
                        else:
                            cliques.append([nid])

            # merge cliques
            for c in range(len(cliques) - 1):
                if c >= len(cliques):
                    break
                for k in range(c + 1, len(cliques)):
                    if k >= len(cliques):
                        break
                    share = list(set(cliques[c]) & set(cliques[k]))
                    if len(share) > 0 and share not in single_cliq:
                        cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                        cliques[k] = []
                cliques = [c for c in cliques if len(c) > 0]
            cliques = [c for c in cliques if len(c) > 0]

            for i, cliq in enumerate(cliques):
                for a in cliq:
                    atom_cliques[a].add(i)  # the value of single link node should be empty{}

            # breaks_all = copy.deepcopy(breaks)
            for b in brics_bonds:
                breaks.append([b[0][0], b[0][1]])

            for b in breaks:
                if not mol.GetAtomWithIdx(b[0]).IsInRing() and mol.GetAtomWithIdx(b[1]).IsInRing():
                    c_idx = atom_cliques[b[0]]
                    if len(c_idx) > 0:  # not single link node
                        cliques[list(c_idx)[0]].append(b[1])
                elif not mol.GetAtomWithIdx(b[1]).IsInRing() and mol.GetAtomWithIdx(b[0]).IsInRing():
                    c_idx = atom_cliques[b[1]]
                    if len(c_idx) > 0:  # not single link node
                        cliques[list(c_idx)[0]].append(b[0])

            for b in breaks:
                if not mol.GetAtomWithIdx(b[0]).IsInRing() and not mol.GetAtomWithIdx(b[1]).IsInRing():
                    cliques.append([b[0], b[1]])
                    if [b[0]] in cliques:
                        cliques.remove([b[0]])
                    if [b[1]] in cliques:
                        cliques.remove([b[1]])

                elif mol.GetAtomWithIdx(b[0]).IsInRing() and mol.GetAtomWithIdx(b[1]).IsInRing():
                    if not ChemUtils.bond_in_ring(mol_bonds, b[0], b[1]):
                        cliques.append([b[0], b[1]])

            for item in single_cliq:
                atom = mol.GetAtomWithIdx(item[0])
                for nei in atom.GetNeighbors():
                    aid = atom.GetIdx()
                    nid = nei.GetIdx()
                    if [nid] in cliques:
                        cliques.remove([nid])
                    if [aid, nid] not in cliques and [nid, aid] not in cliques:
                        if aid < nid:
                            cliques.append([aid, nid])
                        else:
                            cliques.append([nid, aid])

            for i in range(n_atoms):
                atom_cliques[i] = set()

            for i, cliq in enumerate(cliques):
                for a in cliq:
                    atom_cliques[a].add(i)

            single_cliq = []
            for key, value in atom_cliques.items():
                if len(value) >= 3:
                    cliques.append([key])
                    single_cliq.append([key])

            # edges
            edges = []
            singles = set()
            for s in range(len(cliques)):
                s_cliq = cliques[s]
                if len(s_cliq) == 1:
                    singles.add(s)
                    continue
                for e in range(s + 1, len(cliques)):
                    e_cliq = cliques[e]
                    if len(e_cliq) == 1:
                        singles.add(e)
                        continue
                    share = list(set(s_cliq) & set(e_cliq))
                    if len(share) > 0 and share not in single_cliq:
                        edges.append((s, e))

            for i in singles:
                s_cliq = cliques[i]
                for cid in range(len(cliques)):
                    if i == cid:
                        continue
                    share = list(set(cliques[i]) & set(cliques[cid]))
                    if len(share) > 0:
                        if i < cid:
                            edges.append((i, cid))
                        else:
                            edges.append((cid, i))

        except Exception as e:
            print('brics_decomp_extra Exception: ', Chem.MolToSmiles(mol))
            cliques = [list(range(n_atoms))]
            edges = []
            print(e.args)

        return cliques, edges


if __name__ == "__main__":
    print('ChemUtils')
