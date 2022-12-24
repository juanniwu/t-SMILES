
from DataSet.Utils import DSParam
from DataSet.DataGenerator import DataGenerator

from DataSet.STDTokens import STDTokens_SMI_File
from DataSet.STDTokens import EncoderType, CTokens

class DSProvider:
  def data_provider_File(vocab_file, use_cuda=False, batch_size=1, split=0,
                          encoder_type=EncoderType.ET_Int, is_pad=False, startend=True, data_path=None,
                          max_length = 256):

        ctokens = CTokens(STDTokens_SMI_File(vocab_file), 
                          is_pad=is_pad, startend=startend, invalid = True, max_length = max_length)

        print(f'len={len(ctokens.tokens)}, tokens = {ctokens.tokens}')

        rows_to_read = -5
        #rows_to_read = 10

        dsparam = DSParam()
        if data_path is not None:
            dsparam['data_path'] = data_path
        else:
            dsparam['data_path'] = '../RawData/Antiviral/COVID_19/data/all/valid_smiles.smi'

        dsparam['tokens'] = ctokens.tokens
        dsparam['start_token'] = '<'
        dsparam['end_token'] = '>'
        dsparam['max_len'] = ctokens.max_length
        dsparam['use_cuda'] = use_cuda
        dsparam['pad_symbol'] = ' '
        dsparam['seed'] = None
        dsparam['batch_size'] = batch_size
        dsparam['split'] = split
        dsparam['delimiter'] = '\t'  # read_object_property_file
        dsparam['cols_to_read'] = [0]  # read_object_property_file
        dsparam['rows_to_read'] = -1  # read_object_property_file
        dsparam['has_header'] = True  # read_object_property_file

        gen_data = DataGenerator(data_path=dsparam['data_path'],
                                 ctokens=ctokens,
                                 max_len=dsparam['max_len'],
                                 use_cuda=dsparam['use_cuda'],
                                 pad_symbol=dsparam['pad_symbol'],
                                 batch_size=dsparam['batch_size'],
                                 delimiter=dsparam['delimiter'],  # read_object_property_file
                                 cols_to_read=dsparam['cols_to_read'],  # read_object_property_file
                                 rows_to_read=rows_to_read,
                                 has_header=dsparam['has_header'],  # read_object_property_file
                                 encoder_type=encoder_type,
                                 )
        print(f'len(gen_data) = {gen_data.samples_len}, start_token={gen_data.start_token}')
        return {"train": gen_data,
                "val": gen_data,
                "test": gen_data}

