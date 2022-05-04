

def preprocess_text(string:str):
    
    string=string.lower()
    punctuations = '''!()-[]{};:'"\<>/?@#$^&*_~+='''
    string=string.replace('’',"")
    string=string.replace('\n',"")
    for x in string.lower(): 
        if x in punctuations: 
            string = string.replace(x, "") 
    return string
