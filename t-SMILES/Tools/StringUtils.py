import re

class StringUtils:    

    def replae_dummy_id(sml):
        pattern = re.compile(r"\[\d.?\*]|\[\*\]")  #
        finders_a = re.findall(pattern, sml)
        new_str = re.sub(pattern, repl = r'*', string = sml) 
        return new_str 

    