import os
import math
import threading

import time
from datetime import datetime as dt
currentDT = dt.now()
date_str = currentDT.strftime("%Y_%m_%d_%H_%M_%S")

from tqdm import trange

import torch

from tensorboardX import SummaryWriter


from Models.ModelIO import ModelIO

from Tools.Utils import time_since
from Tools.MemUtils import MemUtils

from Models.Base.Trainer import Trainer
from Models.Utils import ModelUtils
from Models.ModelInit import ModelInit


from Logging.LogDataNode import LogDataNode,BCLogger

from MolUtils.RDKUtils.Utils import RDKUtils

from Models.S2S.GPT2 import GPT2, GPT2LanguageModel

#from Models.VAE.VAETrainer import VAETrainer
class GPT2Trainer(Trainer):
    def __init__(self, gen_data, hparam, tparam, model_name = 'GPT2Trainer', *args, **kwargs):
        super(GPT2Trainer, self).__init__(*args, **kwargs)

        self.hparam = hparam
        self.tparam = tparam
        self.device = tparam['device']
        self.m_name = model_name

        self.gen_data = gen_data
        self.gen_data.set_batch_size(hparam['batch_size'])
 
        self.ctoken         = gen_data.ctoken
        self.start_index    = gen_data.start_index
        self.end_index      = gen_data.end_index
        self.padding_idx    = gen_data.pad_index
        
        self.clip           = tparam['clip']

        try:
            self.model = GPT2LanguageModel(hparam = hparam, tparam = tparam, ctoken = self.ctoken)

            self.model.to(self.device)

            self.m_name, self.m_path = Uitls.create_model_name(self.hparam, self.tparam, self.m_name, date_str)
        
            ModelInit.Initlize(self.model, 'xavier_uniform')
        except Exception as e:
            print(e.args)
            self.model = None
            pass
        
        return
  
    def train_step(self, inputs, labels):
        #loss, pred = self.model.train_step(inputs, labels)

        self.model.optimizer.zero_grad()
        outputs = self.model(input_ids = inputs,
                             position_ids = None,
                             token_type_ids = None,
                             past = None,
                             head_mask = None,
                            )              # torch.Size([128, 120, 29])
        pred = outputs[0]
        output_dim = pred.shape[-1]  # 29

        output = pred.contiguous().view(-1, output_dim)     #torch.Size([119, 29])
        #trg = labels[:, 1:].contiguous().view(-1)          #torch.Size([119])
        trg = labels.contiguous().view(-1)                  #torch.Size([120])
            
        loss = self.model.criterion(output, trg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.model.optimizer.step()

        return loss, pred

    def fit(self):
        if self.model is None:
            return None

        model = self.model  #same id(model) = id(self.model)
        gen_data = self.gen_data
 
        tb_log_dir = "./TBSummary/{}_{}/".format(self.m_name, date_str)
        tb_writer = SummaryWriter(log_dir = tb_log_dir)      
        
        log = BCLogger.create_logdata(self.m_name, self.hparam, self.tparam, date_str)
        log.metadata[LogDataNode.key_dataset] = self.gen_data.data_path
        log.metadata[LogDataNode.key_dslen] = self.gen_data.samples_len
        log.metadata[LogDataNode.key_token] = self.gen_data.tokens
        log.metadata[LogDataNode.key_tklen] = self.gen_data.n_characters
        log.metadata[LogDataNode.key_tbsummary] = tb_log_dir
        logfile = log.save(path = self.m_path)
        
        #sub-nodes
        loss_list = []
        train_scores_list = []
        metrics_dict = {}
        scores_list = []

        n_iters = self.tparam['n_iters']
        n_saveinterval = self.tparam['n_saveinterval']

        start = time.time()

        best_model = model.state_dict()
        best_score = -1000
        best_epoch = -1
        terminate_training = False

        n_batches = math.ceil(gen_data.samples_len / gen_data.batch_size)  #335
        n_epochs = n_iters #math.ceil(n_iters/n_batches) #

        losses = 0
        epoch_losses = []
        epoch_scores = []
        id_loss = 0
        id_score = 0
        best_loss = 1000000
        best_score = 0
        best_epoch = 0
        try:
            for epoch in range(n_epochs): #6     
                self.model.train()
                phase = 'train'
                for nb in trange(0, n_batches, desc = f'{phase} in progress...'):
                    inputs, labels = gen_data.random_training_set_batch(batch_size = None, 
                                                                        RNN_Circle = True,
                                                                        return_type = 'long'
                                                                        ) 
                    batch_size, seq_len = inputs.shape[:2]  

                    loss, output = self.train_step(inputs, labels)   
                    losses += loss

                    if self.model.lr_step is not None:
                        self.model.lr_step.step()
                        tb_writer.add_scalar('train_lr', self.model.lr_step.get_last_lr(), epoch)

                    tb_writer.add_scalar('train_loss', loss.item(), id_loss)
                    id_loss += 1

                    #ytrue = labels[:, 1:].cpu().numpy()
                    ytrue = labels.cpu().numpy()
                    ypred = output.cpu().detach().numpy()
                      
                    smils = gen_data.ctoken.onehot_decode(ypred)
                    ypred = gen_data.ctoken.onehot_to_int(ypred)      #(16, 120, 29)   

                    eval_dict={}
                    score = ModelUtils.cls_accuracy(ypred, ytrue) 
                    tb_writer.add_scalar('train_score',score, id_score)
                    id_score += 1
              
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        best_score = score
                        best_epoch = epoch
                        mfile = ModelIO.save_dict(self.model, self.m_path, name = 'best')

                        log.metadata[LogDataNode.key_bestmodel] = mfile
                        log.metadata[LogDataNode.key_bestscore] = [best_epoch, best_loss, best_score]
                        logfile = log.save(path = self.m_path)

                    epoch_losses.append(round(loss.item(),5))
                    epoch_scores.append(round(score,5))

                    loss_list.append(round(loss.item(),5))
                    train_scores_list.append(round(score,5))

                    postfix = f"epoch[{epoch}]-nb[{nb}]"
                    if nb % n_saveinterval == 0 or nb == n_batches - 1:
                        smils_e = self.model.generate_seq(nsamples = self.tparam['n_samples'],
                                                          max_len = self.ctoken.max_length,
                                                          show_atten = False,
                                                          )
                        
                        RDKUtils.SaveSmilesWithImage(smils_e, self.m_path, postfix + '_e')

                        print("\t{}: Epoch={}/{}, batch={}/{}, "
                                "pred_loss={:.4f}, accuracy: {:.2f}, sample: {}".format(time_since(start),
                                                                                        epoch + 1, n_epochs,
                                                                                        nb + 1, n_batches,
                                                                                        loss.item(),
                                                                                        score,
                                                                                        smils[0]
                                                                                        )
                                )
                        mfile = ModelIO.save_dict(self.model, self.m_path, name = postfix)
                
                    MemUtils.rlease_obj(inputs)
                    MemUtils.rlease_obj(labels)
                    MemUtils.rlease_obj(output)
                    MemUtils.rlease_obj(ytrue)
                    MemUtils.rlease_obj(ypred)

                losses = losses / n_batches
                mfile = ModelIO.save_dict(self.model, self.m_path, name = postfix +'_final')

        except Exception as e:
            print('\nException:')
            print(e.args)
            return None

        log.metadata[LogDataNode.key_bestmodel] = mfile
        log.metadata[LogDataNode.key_bestscore] = [best_epoch, best_score]
        log.metadata[LogDataNode.key_final_lr] = self.model.optimizer.param_groups[0]['lr']
        logfile = log.save(path = self.m_path)
                            
        duration = time.time() - start
        print('\nModel training duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))

        tb_writer.close()

        return {'model': model, 
                'score': round(best_loss, 6), #round(np.mean(epoch_scores), 3), 
                'epoch': n_epochs,
                'logfile': logfile}
                
    @torch.no_grad()
    def evaluate(self, log = None, num_smiles = 10000, device = 'cpu', show_atten = False):
        start = time.time()

        self.model.eval()

        samples = []
        step = 10
        count = 0

        for _ in trange(math.ceil(num_smiles / step)):
            smils_e = self.model.generate_seq(nsamples  = step,
                                              max_len   = self.gen_data.ctoken.max_length,
                                              show_atten = show_atten,
                                              )

            samples.extend(smils_e)          
            count += step

        samples = [s.strip() for s in samples]
        res = num_smiles - count
        if res > 0:
            out = self.model.evaluate(step, self.model.zdim)
            samples.append(self.gen_data.ctoken.onehot_decode(out))

        return

class Uitls:
    def create_model_name(hparams, tparams, model_name, date_str):
        m_name = "GPT2[{}-{}]-h[{}]-nh[{}]-nctx[{}]-d[{:.2f}]"\
                "-opt[{}]-lr[{:.5f}]".format(hparams['num_layers'],
                                            hparams['embedding_size'],
                                            hparams['hidden_size'],
                                            hparams['n_heads'],
                                            hparams['n_ctx'],                                                    
                                            hparams['atten_dropout'],                                                    

                                            hparams['optimizer'],
                                            hparams['optimizer_lr'],
                                            )
        m_path = os.path.join('./model_dir/', model_name, m_name, date_str)
        os.makedirs(m_path, exist_ok = True)      
        print(m_path)
        return m_name, m_path

