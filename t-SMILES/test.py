from DataSet.STDTokens import CTokens, STDTokens_Frag_File

from MolUtils.RDKUtils.Frag.RDKFragUtil import Fragment_Alg
from DataSet.Graph.CNJMolAssembler import CNJMolAssembler
from DataSet.Graph.CNJMolUtil import CNJMolUtil           
from DataSet.Graph.CNJTMol import CNJMolUtils

from DataSet.Graph.CNJMolAssembler import rebuild_file
from DataSet.Graph.CNJTMol import preprocess



def test_encode():
    smls = 'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'  #celecoxib

    dec_algs = [
     Fragment_Alg.Vanilla,
     Fragment_Alg.JTVAE,
     Fragment_Alg.BRICS,
     Fragment_Alg.BRICS_Base,
     Fragment_Alg.MMPA,
     Fragment_Alg.Scaffold,
     Fragment_Alg.BRICS_DY,
     Fragment_Alg.MMPA_DY,
     Fragment_Alg.Scaffold_DY,
     #Fragment_Alg.RBrics_DY,
    ]

    print('[smls is]:', smls)

    ctoken = CTokens(STDTokens_Frag_File(None), max_length = 256, invalid = True, onehot = False)


    for dec_alg in dec_algs:
        combine_sml, combine_smt = CNJMolUtils.encode_single(smls, ctoken, dec_alg)
    
        print('[dec_alg is]:', dec_alg.name)
        print('[TSSA/TSDY]:', combine_sml)  
        print('[TSID     ]:', combine_smt)     
   
    return 


def test_decode():
    maxlen = 512
    vocab_file = r'../RawData/Chembl/Test/Chembl_test.smi.[MMPA_DY][237]_token.voc.smi'

    ctoken = CTokens(STDTokens_Frag_File(vocab_file), is_pad = True, pad_symbol = ' ', startend = True,
                     max_length = maxlen,  flip = False, invalid = True, onehot = False)

    #SMILES	 = 'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F' #Celecoxib

    #bfs_ex = 'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F&&&'	#TS_Vanilla	

    #bfs_ex = 'CC&C1=CC=CC=C1&CC&C1=C[NH]N=C1&CC&C^CN^CF&C1=CC=CC=C1&&CF^CS&&CF^S&&&NS&&O=S&O=S&&'	#TSSA-J	
    #bfs_ex = 'CC1=CC=C(C2=CC(C(F)(F)F)=NN2C2=CC=C(S(N)(=O)=O)C=C2)C=C1&&&'	#TSSA-B	
    #bfs_ex = 'CC&C1=CC=CC=C1&CC&C1=C[NH]N=C1&CN&C1=CC=CC=C1^CC^CS&C&N[SH](=O)=O&CF&&&&FCF&&'	#TSSA-M	
    #bfs_ex = 'CC&C1=CC=C(C2=CC=NN2C2=CC=CC=C2)C=C1&CC&FC(F)F^CS&&N[SH](=O)=O&&&'	    #TSSA-S	

    #bfs_ex = 'CC1=CC=C(C2=CC(C(F)(F)F)=NN2C2=CC=C(S(N)(=O)=O)C=C2)C=C1&&&'	#TSDY-B
    #bfs_ex = '*C&*C1=CC=C(*)C=C1&*C1=CC(*)=NN1*&*C(*)(F)F&*F^*C1=CC=C(*)C=C1&&*S(N)(=O)=O&&&'	#TSDY-M	
    #bfs_ex = '*C&*C1=CC=C(C2=CC(*)=NN2C2=CC=C(*)C=C2)C=C1&*C(F)(F)F&&*S(N)(=O)=O&&'	#TSDY-S	

    #bfs_ex = 'CC1=CC=C(C2=CC(C(F)(F)F)=NN2C2=CC=C(S(N)(=O)=O)C=C2)C=C1&&&'	#TSID-B	
    #bfs_ex = '[1*]C&[1*]C1=CC=C([2*])C=C1&[2*]C1=CC([3*])=NN1[5*]&[3*]C([4*])(F)F&[4*]F^[5*]C1=CC=C([6*])C=C1&&[6*]S(N)(=O)=O&&&'	#TSID_M	
    #bfs_ex = '[1*]C&[1*]C1=CC=C(C2=CC([2*])=NN2C2=CC=C([3*])C=C2)C=C1&[2*]C(F)(F)F&&[3*]S(N)(=O)=O&&'	#TSID-S	


    #bfs_ex = 'O=[PH](O)O&CP&C&CN&CN^CC^CC&C1=CC=CC=C1&C1=CC=CC=C1&&&&&'
    #bfs_ex = 'CCc1ccc(CCP)cc1&O=[PH](O)O&&&'
    
    #asm_alg = 'CALG_TSSA'    
    asm_alg = 'CALG_TSDY'    
    #asm_alg = 'CALG_TSID'    

    bfs_ex = ''.join(bfs_ex.strip().split(' '))
    print('input:=', bfs_ex)


    bfs_ex_smiles = CNJMolUtil.split_ex_smiles(bfs_ex, delimiter='^')
    print('bfs_ex_smiles', bfs_ex_smiles)     
    
    n_samples = 2
    for i in range(n_samples):
        re_smils, bfs_ex_smiles_sub, new_vocs_sub = CNJMolAssembler.decode_single(bfs_ex, ctoken , asm_alg, n_samples = 1, p_mean = None) 
        print('dec_smile:=', re_smils)
    
    return 


if __name__ == '__main__':

    #test_encode()

    #test_decode()

    preprocess()

    #rebuild_file()
