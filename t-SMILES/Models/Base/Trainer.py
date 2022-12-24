from Tools.MemUtils import MemUtils

class Trainer(object):
    def __init__(self, *args, **kwargs):
        super(Trainer, self).__init__()#*args, **kwargs

    
    def __del__(self):
        MemUtils.rlease_obj(self.model)
        return

    def fit(self, *args, **kwargs):
        pass

    def evaluate(self, *args, **kwargs):
        pass

        

 
