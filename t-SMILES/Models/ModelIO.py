import os
import torch

class ModelIO: 
    @staticmethod#(Recommended)
    def save_dict(model, path, name):
        os.makedirs(path, exist_ok = True)
        mfile = os.path.join(path, name + '.dict')

        if isinstance(model, dict):
            save_data = {}
            for key, value in model.items():
                save_data[key] = value.state_dict()
        else:
            save_data = model.state_dict()

        torch.save(save_data, mfile)
        return mfile

    @staticmethod#(Recommended)
    def load_dict(model, mfile, device = 'cpu', combined = False):
        if os.path.isfile(mfile):
            sm = torch.load(mfile, map_location = device)
            #if isinstance(sm, dict):
            if combined:   #several models in one file
                for key, value in sm.items():
                    model[key].load_state_dict(value, strict = True)
            else:
                model.load_state_dict(sm, strict = True)            

        return model
      
    @staticmethod
    def save_whole_model(model, path, name):
        os.makedirs(path, exist_ok = True)
        mfile = os.path.join(path, name + '.mod')

        torch.save(model, mfile)
        return mfile

