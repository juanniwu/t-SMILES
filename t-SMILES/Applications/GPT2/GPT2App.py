import os
print(os.getcwd())

from datetime import datetime as dt
currentDT = dt.now()
print(currentDT)
date_str = currentDT.strftime("%Y_%m_%d_%H_%M_%S")

import sys
sys.path.append('./DataSet')
sys.path.append('./OpenChem')
sys.path.append('./Vectorization')
sys.path.append('./Visualization')
sys.path.append('./Models')
sys.path.append('./MolUtils')
sys.path.append('./Applications')
sys.path.append('./Math')
sys.path.append('./Tools')
sys.path.append('./HParam')
sys.path.append('./Logging')
sys.path.append('./ptan')

import torch
import pandas as pd
from tqdm import tqdm, trange

if torch.cuda.is_available():
    dvc_id = 0
    use_cuda = True
    device = f'cuda:{dvc_id}'
    torch.cuda.set_device(dvc_id)
else:
    device = 'cpu'
    use_cuda = False
    
device = 'cpu'
use_cuda = False

from DataSet.DataProvider import DSProvider
from DataSet.STDTokens import EncoderType
from Models.Parameters import TParam
from Models.ModelIO import ModelIO

from Logging.LogDataNode import LogDataNode, JSONParse, BCLogger

from Models.S2S.GPT2 import GPT2_HParam
from Models.S2S.GPT2Trainer import GPT2Trainer

def HParam_5_512(hparam):
    hparam['batch_size'] = 256
    
    hparam['optimizer'] = 'Adam'
    hparam['optimizer_lr'] =  0.001
    hparam['optimizer_weight_decay'] = 1e-4  

    hparam['num_layers'] = 5  
    hparam['embedding_size'] = 512
    hparam['hidden_size'] = 512

    hparam['n_heads'] = 8
    hparam['n_ctx'] = 256
    hparam['layer_norm_epsilon'] = 1e-05

    hparam['embd_pdrop'] = 0.1624
    hparam['atten_dropout'] = 0.2687
    hparam['resid_dropout'] = 0.1467
   
    hparam['initializer_range'] = 0.02
    hparam['torchscript'] = False

    hparam['optimizer'] =  'Adam'
    hparam['optimizer_lr'] = 0.0001
    hparam['optimizer_weight_decay'] = 0.0001

    return hparam

def train_single_voc_file():
    hparam = GPT2_HParam()
    hparam = HParam_5_512(hparam)
    hparam['batch_size'] = 4

    tparam = TParam()
    tparam['device'] = device
    tparam['use_cuda'] = use_cuda
    tparam['n_iters'] = 1
    tparam['n_saveinterval'] = 1
    tparam['n_samples'] = 10
    tparam['lr_step'] = True
    tparam['lr_step_size'] = 5*12078    #10*3096
    tparam['clip'] = 1

    #vocab_file1 ='../RawData/Antiviral/coronavirus_data/data/AID1706/AID1706.smi.tokens_single.csv' 
    #vocab_file2 ='../RawData/Antiviral/coronavirus_data/data/AID1706/AID1706.smi.bfs[132]_smiles_join.csv.tokens_single.csv' 
    #src_file = None
    #src_file = '../RawData/Antiviral/COVID_19/data/all/valid_smiles.smi'
    #src_file = '../RawData/ChEMBL/chembl_21_1576904.csv_Canonical.smi.bfs_ex_vocids[228]_smiles_join.csv'
    #src_file = '../RawData/ChEMBL/chembl_21_1576904.csv_Canonical.smi.bfs[228]_smiles_join.csv'
    #src_file = '../RawData/Antiviral/coronavirus_data/data/AID1706/AID1706.smi'


    #vocab_file1 ='../RawData/ChEMBL/chembl_21.csv_Canonical.smi.[100_200]bfs[356]_org.csv.tokens_single.csv' 
    #vocab_file2 ='../RawData/ChEMBL/chembl_21.csv_Canonical.smi.[100_200]bfs[356]_org.csv.tokens.csv' 
    #vocab_file3 ='../RawData/ChEMBL/chembl_21.csv_Canonical.smi.[100_200]bfs[356]_smiles_join.csv.tokens_single.csv' 
    #vocab_file4 ='../RawData/ChEMBL/chembl_21.csv_Canonical.smi.[100_200]bfs[356]_smiles_join.csv.tokens.csv' 
    #src_file ='../RawData/ChEMBL/chembl_21.csv_Canonical.smi.[100_200]bfs[356]_smiles_join.csv' 
    #src_file ='../RawData/ChEMBL/chembl_21.csv_Canonical.smi.[100_200]bfs[356]_org.smi' 
    
    #vocab_file1 ='../RawData/ChEMBL/chembl_120_Canonical.smi.bfs[228]_org.smi.tokens_single.csv' 
    #vocab_file2 ='../RawData/ChEMBL/chembl_120_Canonical.smi.bfs[228]_org.smi.tokens.csv' 
    #vocab_file3 ='../RawData/ChEMBL/chembl_120_Canonical.smi.bfs[228]_smiles_join.csv.tokens_single.csv' 
    #vocab_file4 ='../RawData/ChEMBL/chembl_120_Canonical.smi.bfs[228]_smiles_join.csv.tokens.csv' 
    #src_file ='../RawData/ChEMBL/chembl_120_Canonical.smi.bfs[228]_smiles_join.csv' 
    #src_file ='../RawData/ChEMBL/chembl_120_Canonical.smi.bfs[228]_org.smi' 

    #------------------------------
    #vocab_file1 ='../RawData/JTVAE/data/zinc/Brics_bridge/all.txt.[BRICS][64]_join.csv.tokens_single.csv' 
    #vocab_file2 ='../RawData/JTVAE/data/zinc/Brics_bridge/all.txt.[BRICS][64]_join.csv.tokens.csv' 
 
    #vocab_file1 ='../RawData/JTVAE/data/zinc/Brics/all.txt.[BRICS][64]_join.csv.tokens_single.csv' 
    #vocab_file2 ='../RawData/JTVAE/data/zinc/Brics/all.txt.[BRICS][64]_join.csv.tokens.csv' 

    #vocab_file3 ='../RawData/JTVAE/data/zinc/all.txt.bfs[64]_org.csv.tokens_single[Moses].csv' 
    #vocab_file1 ='../RawData/JTVAE/data/zinc/all.txt.bfs[64]_smiles_join.csv.tokens_single.csv' 
    #vocab_file2 ='../RawData/JTVAE/data/zinc/all.txt.bfs[64]_smiles_join.csv.tokens.csv' 
    #vocab_file1 ='../RawData/JTVAE/data/zinc/all.txt.bfs[64]_org.csv.tokens_single.csv' 
    vocab_file2 ='../RawData/JTVAE/data/zinc/all.txt.bfs[64]_org.csv.tokens.csv' 

    #src_file ='../RawData/JTVAE/data/zinc/Brics/all.txt.[BRICS][64]_join.csv' 
    #src_file ='../RawData/JTVAE/data/zinc/all.txt.bfs[64]_org.smi.bfs[56]_brics_join.csv' 
    #src_file ='../RawData/JTVAE/data/zinc/all.txt.bfs[64]_smiles_join.csv' 
    src_file ='../RawData/JTVAE/data/zinc/all.txt.bfs[64]_org.smi' 
    #src_file ='../RawData/JTVAE/data/zinc/Brics_bridge/all.txt.[BRICS][64]_join.csv' 
   
    #------------------------------
    #vocab_file1 ='../RawData/QM/QM9/frag_token.csv' 
    #vocab_file1 ='../RawData/QM/QM9/QM9.smi.bfs[24]_smiles_join.csv.tokens_single.csv' 
    #vocab_file2 ='../RawData/QM/QM9/QM9.smi.bfs[24]_smiles_join.csv.tokens.csv' 
    #vocab_file3 ='../RawData/QM/QM9/QM9.smi.tokens_single.csv' 
    #vocab_file4 ='../RawData/QM/QM9/QM9.smi.tokens.csv' 
    #src_file ='../RawData/QM/QM9/QM9.smi.bfs[24]_smiles_join.csv' 
    #src_file ='../RawData/QM/QM9/qm9.smi' 

    #vocab_file1 ='../RawData/Antiviral/coronavirus_data/data/AID1706/AID1706.smi.tokens_single.csv' 
    #vocab_file2 ='../RawData/Antiviral/coronavirus_data/data/AID1706/AID1706.smi.bfs[132]_smiles_join.csv.tokens_single.csv' 
    #src_file = '../RawData/Antiviral/coronavirus_data/data/AID1706/AID1706.smi.bfs[132]_smiles_join.csv'
   
    #src_file = '../RawData/Antiviral/COVID_19/data/all/valid_smiles.smi'
    #src_file = '../RawData/ChEMBL/chembl_21_1576904.csv_Canonical.smi.bfs_ex_vocids[228]_smiles_join.csv'
    #src_file = '../RawData/ChEMBL/chembl_21_1576904.csv_Canonical.smi.bfs[228]_smiles_join.csv'
    #src_file = '../RawData/Antiviral/coronavirus_data/data/AID1706/AID1706.smi'
      
    #src_file = None

    tparam['vocab_file'] = [vocab_file2]
    tparam['src_file'] = src_file
    tparam['max_length'] = 120
 
    gen_data = DSProvider.data_provider_File(vocab_file = [vocab_file2],
                                              batch_size = hparam['batch_size'],
                                              use_cuda = use_cuda, is_pad = True, startend = True,
                                              encoder_type = EncoderType.ET_Int,
                                              data_path = src_file,
                                              max_length = tparam['max_length'],
                                              )['train']

    if len(gen_data.samples) == 0:
        print('data loading error!!!')
        return 

    in_channels = gen_data.ctoken.max_length   #120
    vocab_size = len(gen_data.all_characters)  #29
    in_features = vocab_size 
  
    tparam['in_channels'] = in_channels
    tparam['in_features'] = in_features
    tparam['out_features'] = in_features
    hparam['n_ctx'] = in_channels

    trainer = GPT2Trainer(gen_data, hparam, tparam, model_name = 'GPT2_Brics[RRB Break]_Zinc')

    #init_model = r'H:\BioChemoTCH_Target\model_dir\S2S\GPT2_Zinc\GPT2[3-512]-h[256]-nh[8]-nctx[1024]-d[0.1]-opt[Adam]-lr[0.00050]\2021_07_28_09_21_40\best.dict'
    #init_model = r'H:\BioChemoTCH_Target\model_dir\S2S\GPT2_Zinc\GPT2[5-512]-h[512]-nh[4]-nctx[256]-d[0.2687]-opt[Adam]-lr[0.00010]\2021_09_26_09_27_05\best.dict'
   
    #init_model = r'H:\BioChemoTCH_Target\model_dir\Frag\GPT2_Frag_Double_Zinc\GPT2[5-512]-h[512]-nh[8]-nctx[256]-d[0.27]-opt[Adam]-lr[0.00010]\2022_06_02_23_46_31\best.dict'
    #init_model = r'H:\BioChemoTCH_Target\model_dir\Frag\GPT2_Frag_Double_QM9\GPT2[3-512]-h[512]-nh[8]-nctx[256]-d[0.27]-opt[Adam]-lr[0.00010]\2022_05_27_19_23_28\best.dict'
    #init_model = r'H:\BioChemoTCH_Target\model_dir\Frag\GPT2_SML_Double_Zinc\GPT2[5-512]-h[512]-nh[8]-nctx[120]-d[0.27]-opt[Adam]-lr[0.00010]\2022_07_06_08_18_40\best.dict'
    #init_model = r'H:\BioChemoTCH_Target\model_dir\Frag_Brics\GPT2_Brics[RRB_Break]_MTP_Zinc\GPT2[5-512]-h[512]-nh[8]-nctx[256]-d[0.27]-opt[Adam]-lr[0.00010]\2022_07_15_05_59_46\best.dict'

    #tparam['init_model'] = init_model

    #if init_model is not None:
    #    ModelIO.load_dict(trainer.model, init_model, tparam['device'])
      
    result = trainer.fit()

    #result =  trainer.evaluate(num_smiles = 500, show_atten = True)

    return result


def generate_seq():
    #JTVAE
    #init_model = r'D:\ProjectTF\BioChemoTCH_2022_0517_Frag\model_dir\GPT2_Frag_Double_QM9\GPT2[3-512]-h[512]-nh[8]-nctx[256]-d[0.27]-opt[Adam]-lr[0.00010]\2022_05_27_19_23_28\best.dict'
    #dict_cfg   = r'H:\BioChemoTCH_Target\model_dir\Frag\GPT2_Frag_Double_QM9\GPT2[3-512]-h[512]-nh[8]-nctx[256]-d[0.27]-opt[Adam]-lr[0.00010]\2022_05_27_19_23_28\model.json'
    #src_file   = r'H:\BioChemoTCH_Target\t-SMILES\Data\QM9\QM9.smi.bfs[24]_smiles_join.csv'

    init_model  = r'K:\BioChemoTCH_Target\Frag\GPT2_Frag_Double_Zinc\GPT2[5-512]-h[512]-nh[8]-nctx[256]-d[0.27]-opt[Adam]-lr[0.00010]\2022_06_02_23_46_31[Vocs=37]\best.dict'
    dict_cfg    = r'K:\BioChemoTCH_Target\Frag\GPT2_Frag_Double_Zinc\GPT2[5-512]-h[512]-nh[8]-nctx[256]-d[0.27]-opt[Adam]-lr[0.00010]\2022_06_02_23_46_31[Vocs=37]\model.json'
    src_file    = r'H:\BioChemoTCH_Target\t-SMILES\Data\Zinc\JTVAE\all.txt.bfs[64]_smiles_join.csv'

    #init_model  = r'H:\BioChemoTCH_Target\model_dir\Frag\GPT2_Frag_Chembl\GPT2[5-512]-h[512]-nh[8]-nctx[256]-d[0.27]-opt[Adam]-lr[0.00010]\2022_05_31_10_43_19\best.dict'
    #dict_cfg    = r'H:\BioChemoTCH_Target\model_dir\Frag\GPT2_Frag_Chembl\GPT2[5-512]-h[512]-nh[8]-nctx[256]-d[0.27]-opt[Adam]-lr[0.00010]\2022_05_31_10_43_19\model.json'
    #src_file    = r'H:\BioChemoTCH_Target\t-SMILES\Data\Chembl\JTVAE\chembl_120_Canonical.smi.bfs[228]_smiles_join.csv'

    #brics_chembl
    #init_model  = r'H:\BioChemoTCH_Target\model_dir\Frag_Brics\GPT2_Brics[RRB_Break]_MTP_Chembl\GPT2[5-512]-h[512]-nh[8]-nctx[256]-d[0.27]-opt[Adam]-lr[0.00010]\2022_07_17_19_15_42\best.dict'
    #dict_cfg    = r'H:\BioChemoTCH_Target\model_dir\Frag_Brics\GPT2_Brics[RRB_Break]_MTP_Chembl\GPT2[5-512]-h[512]-nh[8]-nctx[256]-d[0.27]-opt[Adam]-lr[0.00010]\2022_07_17_19_15_42\model.json'
    #src_file    = r'H:\BioChemoTCH_Target\t-SMILES\Data\Chembl\Brics_bridge\chembl_120.bfs[228]_org.smi.[BRICS][228]_join.csv'

    #init_model  = r'H:\BioChemoTCH_Target\model_dir\Frag_Brics\GPT2_Brics[RRB_Break]_MTP_Zinc\GPT2[5-512]-h[512]-nh[8]-nctx[256]-d[0.27]-opt[Adam]-lr[0.00010]\2022_07_15_05_59_46\best.dict'
    #dict_cfg    = r'H:\BioChemoTCH_Target\model_dir\Frag_Brics\GPT2_Brics[RRB_Break]_MTP_Zinc\GPT2[5-512]-h[512]-nh[8]-nctx[256]-d[0.27]-opt[Adam]-lr[0.00010]\2022_07_15_05_59_46\model.json'
    #src_file    = r'H:\BioChemoTCH_Target\t-SMILES\Data\Zinc\Brics_bridge\all.txt.[BRICS][64]_join.csv'

    #smiles
    #init_model  = r'H:\BioChemoTCH_Target\model_dir\Frag\GPT2_SML_Double_Chembl\GPT2[5-512]-h[512]-nh[8]-nctx[256]-d[0.27]-opt[Adam]-lr[0.00010]\2022_06_06_19_13_06\best.dict'
    #dict_cfg    = r'H:\BioChemoTCH_Target\model_dir\Frag\GPT2_SML_Double_Chembl\GPT2[5-512]-h[512]-nh[8]-nctx[256]-d[0.27]-opt[Adam]-lr[0.00010]\2022_06_06_19_13_06\model.json'
    #src_file    = r'H:\BioChemoTCH_Target\t-SMILES\Data\Chembl\chembl_120_Canonical.smi.bfs[228]_org.smi'

    #init_model  = r'H:\BioChemoTCH_Target\model_dir\Frag\GPT2_SML_Double_Zinc\GPT2[5-512]-h[512]-nh[8]-nctx[120]-d[0.27]-opt[Adam]-lr[0.00010]\2022_07_06_08_18_40\best.dict'
    #dict_cfg    = r'H:\BioChemoTCH_Target\model_dir\Frag\GPT2_SML_Double_Zinc\GPT2[5-512]-h[512]-nh[8]-nctx[120]-d[0.27]-opt[Adam]-lr[0.00010]\2022_07_06_08_18_40\model.json'
    #src_file    = r'H:\BioChemoTCH_Target\t-SMILES\Data\Zinc\all.txt.bfs[64]_org.smi'

    #init_model  = r'H:\BioChemoTCH_Target\model_dir\Frag\GPT2_SML_Double_QM9\GPT2[5-512]-h[512]-nh[8]-nctx[256]-d[0.27]-opt[Adam]-lr[0.00010]\2022_07_07_10_47_31\best.dict'
    #dict_cfg    = r'H:\BioChemoTCH_Target\model_dir\Frag\GPT2_SML_Double_QM9\GPT2[5-512]-h[512]-nh[8]-nctx[256]-d[0.27]-opt[Adam]-lr[0.00010]\2022_07_07_10_47_31\model.json'
    #src_file    = r'H:\BioChemoTCH_Target\t-SMILES\Data\QM9\QM9.smi.bfs[24]_org.csv'

    metadata = JSONParse.read_item(dict_cfg,'metadata')
    hparam = JSONParse.read_item(dict_cfg,'hparam')
    tparam =  JSONParse.read_item(dict_cfg,'tparam')

    if 'max_length' not in tparam:
        tparam['max_length'] = 120
    tparam['src_file'] = src_file

    token = metadata['token']
    token.remove(' ')
    vocab_file = dict_cfg+ '_voc.tmp.txt'

    output = vocab_file
    df = pd.DataFrame(token)
    df.to_csv(output, index = False, header=False, na_rep="NULL")

    gen_data = DSProvider.data_provider_File(vocab_file = [vocab_file],
                                              batch_size = hparam['batch_size'],
                                              use_cuda = use_cuda, is_pad = True, startend = True,
                                              encoder_type = EncoderType.ET_Int,
                                              data_path = src_file,
                                              max_length = tparam['max_length'],
                                              )['train']

    ctk = gen_data.ctoken.tokens

    trainer = GPT2Trainer(gen_data, hparam, tparam, model_name = 'GPT2_Remodel')
    if init_model is not None:
        ModelIO.load_dict(trainer.model, init_model, tparam['device'])
 
    model = trainer.model

    model.eval()
    n_samples = 100

    samples = []

    for _ in trange(n_samples):
        smils_e = model.generate_seq(nsamples  = 1,
                                    max_len   = gen_data.ctoken.max_length,
                                    show_atten = False,
                                    )
        smils_e = smils_e[-1]
        print(smils_e)
        samples.append(smils_e)

    output = os.path.join(path, 'generated.csv')
    df = pd.DataFrame(samples)
    df.to_csv(output, index = False, header=False, na_rep="NULL")

    return

if __name__ == '__main__':

    train_single_voc_file()

    #generate_seq()  


