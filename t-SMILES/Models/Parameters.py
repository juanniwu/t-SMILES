
def TParam():
    config = {        
        'input_size'    : 46,
        'output_size'   : 46,
        'n_iters'       : 1500000,
        'n_saveinterval': 100,
        'n_samples'     : 100,
        'device'        : 'cpu',
        'use_cuda'      : False,
        'batch_first'   : False,
     
        'lr_step'       : True,
        'lr_step_size'  : 100,
        'lr_step_gamma' : 0.5,
        'clip_grad'    : -50,
    }
    return config
