#reference:https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/21
import torch
import gc
#import resource   #pip install python-resources
import multiprocessing as mp

#from Tools.MemUtils import MemUtils
class MemUtils:

    def rlease_obj(obj):
        if obj is not None:
            del obj
        gc.collect()
        torch.cuda.empty_cache()

    def release_optimizer(optimizer):
        if optimizer is None:
            return

        MemUtils.optimizer_to(optimizer, torch.device('cpu'))  #def wipe_memory(self): # DOES WORK
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()

    def optimizer_to(optimizer, device):
        for param in optimizer.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

