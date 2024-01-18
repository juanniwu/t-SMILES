
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

