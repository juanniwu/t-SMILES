import numpy as np
import torch
import torch.optim as optim
import sklearn.metrics as metrics

class ModelUtils:
    def cls_accuracy(y_true, y_pred):
        acc = 0
        if isinstance(y_pred, (list, tuple, np.ndarray)):
            if isinstance(y_pred[0], (list, tuple, np.ndarray)): 
                for i in range(len(y_true)):
                    yt = y_true[i]
                    yp = y_pred[i]
                    acc += metrics.accuracy_score(yt, yp)
                acc /= len(y_true)
            else:
                acc = metrics.accuracy_score(y_true, y_pred)
        return acc

    def create_optimizer(model, hparam, tparam):
        #opt_param = ModelUtils.parse_hparam(hparam)
        optimizer = ModelUtils.parse_optimizer(hparam, model)

        #lr_scheduler = tparam['lr_scheduler']
        lr_scheduler = 'StepLR'
        if tparam['lr_step']:
            if lr_scheduler == 'CosineAnnealingLR':
                # here we do not set eta_min to lr_min to be backward compatible
                # because in previous versions eta_min is default to 0
                # rather than the default value of lr_min 1e-6
                lr_step = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max = tparam['max_step'],
                                                               eta_min = tparam['eta_min'])  # should use eta_min arg
            elif lr_scheduler == 'LambdaLR':
                # originally used for Transformer (in Attention is all you need)
                def lr_lambda(step):
                    # return a multiplier instead of a learning rate
                    if step == 0 and tparam['warmup_step'] == 0:
                        return 1.
                    else:
                        return 1. / (step ** 0.5) if step > tparam['warmup_step'] else step / (tparam['warmup_step'] ** 1.5)

                lr_step = optim.lr_scheduler.LambdaLR(optimizer, 
                                                      lr_lambda = lr_lambda)
            elif lr_scheduler == 'ReduceLROnPlateau':
                lr_step = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor = tparam['decay_rate'] ,
                                                               patience = tparam['patience'] ,
                                                               min_lr = tparam['lr_min'] 
                                                               )
            elif lr_scheduler == 'MultiplicativeLR':
                pass
            elif lr_scheduler == 'MultiStepLR':
                pass
            elif lr_scheduler == 'ExponentialLR':
                pass
            elif lr_scheduler == 'CyclicLR':
                pass
            elif lr_scheduler == 'OneCycleLR':
                pass
            elif lr_scheduler == 'constant':
                pass
            elif lr_scheduler == 'CosineAnnealingWarmRestarts':
                pass
            else:  #StepLR
                lr_step = optim.lr_scheduler.StepLR(optimizer ,
                                                tparam['lr_step_size'],
                                                tparam['lr_step_gamma']
                                                )
        else:
            lr_step = None

        return optimizer, lr_step


    def parse_optimizer(hparam, model, **kwargs):
        optimizer = {
            'adam'      : torch.optim.Adam,
        }.get(hparam['optimizer'].lower(), None)

        if isinstance(model, (list, set, tuple)):
            params = list()
            for m in model:
                params += list(m.parameters())
        else:
            params = model.parameters()

        sample_softmax = 0
        if sample_softmax > 0:
            dense_params, sparse_params = [], []
            for param in model.parameters():
                if param.size() == model.word_emb.weight.size():
                    sparse_params.append(param)
                else:
                    dense_params.append(param)

            optimizer = optim.SGD(sparse_params, lr = hparam['optimizer_lr'] * 2)
            optimizer = optim.SparseAdam(sparse_params, lr = hparam['optimizer_lr'])
            #optimizer = optim.Adam(dense_params, lr=opt_param['lr'])
        else:
            if optimizer is None:
                optimizer = torch.optim.Adam

            optim_kwargs = dict()
            optim_kwargs['lr'] = hparam['optimizer_lr']
            optim_kwargs['weight_decay'] = hparam['optimizer_weight_decay']

            if hparam['optimizer'].lower() == 'sgd' or hparam['optimizer'].lower() == 'rmsprop':
                optim_kwargs['momentum'] = hparam['opt_momentum']

            if hparam['optimizer'].lower() == 'rmsprop':
                optim_kwargs['alpha'] = hparam['opt_alpha']

            # optimizer = optimizer(model.parameters(), **optim_kwargs)
            optimizer = optimizer(params, **optim_kwargs)
        return optimizer



