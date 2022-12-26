import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scipy
from scipy import histogram
from scipy.stats import entropy, gaussian_kde
from functools import singledispatch
import operator

from sklearn.preprocessing import normalize
from sklearn import preprocessing


#from Tools.MathUtils import BCMathUtils
class BCMathUtils:
    def find_index(datalist, node):
        datalist = list(datalist)
        index = -1
        for i in range(len(datalist)):
            if datalist[i] == node:
                index = i
                break
        return index

    def dict_sort_key(dictdata, reverse=False):
        keys = dictdata.keys()
        keys = sorted(keys, reverse = reverse)
        res = {}
        for key in keys:
            res[key] = dictdata[key]
        return res

