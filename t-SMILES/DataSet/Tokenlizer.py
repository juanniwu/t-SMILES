
from Tools.MathUtils import BCMathUtils

class Tokenlizer():
    def __init__(self, voc, invalid_token = '&'):
        self.voc = voc
        self.invalid_token = invalid_token

        self.fdict = {}
        for tk in voc:
            slen = str(len(tk))
            if  slen in self.fdict.keys():
                self.fdict[slen].append(tk)
            else:
                self.fdict[slen] = ['']
                self.fdict[slen].append(tk)

        self.fdict = BCMathUtils.dict_sort_key(self.fdict, reverse = True)

        self.max_len = max([int(key) for key in self.fdict.keys()])

        return

    def tokenize(self, inseq):
        tokens = TokenlizerUtil.forward_segment(text = inseq, dic = self.voc)

        return tokens
           

class TokenlizerUtil:
    def forward_segment(text, dic):
        word_list = []
        i = 0
        while i < len(text):
            longest_word = text[i]                     
            for j in range(i + 1, len(text) + 1):       
                word = text[i:j]                        
                if word in dic:                         
                    if len(word) > len(longest_word):  
                        longest_word = word            
            word_list.append(longest_word)             
            i += len(longest_word)

        return word_list


