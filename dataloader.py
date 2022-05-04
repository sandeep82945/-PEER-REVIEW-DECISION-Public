
import itertools
import pickle
from utils import preprocess_text
import torch
import os
import random
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from vader import get_score
import numpy as np
from sentence_transformers import SentenceTransformer


torch.manual_seed(400)
np.random.seed(400)

torch.cuda.set_device(1)
model=SentenceTransformer('stsb-roberta-base')
os.environ['TOKENIZERS_PARALLELISM']='False'

def Jaccard_Similarity(doc1, doc2): 
    #Set the threshold
    t = 0.55
    # List the unique words in a document
    words_doc1 = set(doc1.lower().split()) 
    words_doc2 = set(doc2.lower().split())
    
    # Find the intersection of words list of doc1 & doc2
    intersection = words_doc1.intersection(words_doc2)

    # Find the union of words list of doc1 & doc2
    union = words_doc1
        
    # Calculate Jaccard similarity score 
    # using length of intersection set divided by length of union set
    score = float(len(intersection)) / len(union)
    if(score > t):
        return True
    else:
        return False    
            
class Data(Dataset):
    def __init__(self,ids,papers,labels,ratingss):

        self.ids=ids
        self.papers=papers
        self.labels=labels
        self.scores=ratingss
        self.max_len=50
        self.size=len(ids)

    @classmethod
    def getReader(cls,low,up,conf_name):
        with open("input_files/paper_review_phrases_JCDL_8"+conf_name+".pickle",'rb') as out:
            paper_files=pickle.load(out)
            a=list(paper_files.keys())[low:up]
            b=list(paper_files.values())[low:up]
            
            paper_files={k:v for k,v in zip(a,b)}
            paper_ids=list(paper_files.keys())
        
        with open("input_files/paper_decision_JCDL_8"+conf_name+".pickle",'rb') as out:
            paper_decision=pickle.load(out)
            d=[paper_decision[k] for k in paper_ids]

            print("paper accepted-",d.count('accept'))
            print("paper rejected-",d.count('reject'))
            print()
        
        with open("input_files/paper_rating_JCDL_8"+conf_name+".pickle",'rb') as out:
            paper_rating=pickle.load(out)

        return cls(paper_ids,paper_files, paper_decision,paper_rating)

    def __getitem__(self,idx):
        var=150
        aspect={'motivation':0,'clarity':1,'substance':2,'soundness':3,'meaningful_comparison':4,'originality':5,'replicability':6,'no_aspect':7}
        sentiment={'positive':0,'negative':1,'no_sentiment':2}
        deci={'accept':1,'reject':0}
 
        p_id=self.ids[idx]
        score_sen=[]
        r_sen=[]
        rating_data=self.scores[p_id]
        aspect_data=self.papers[p_id]
           
        for key,values in aspect_data.items():
            r_sen.append(values)
            
            for key1 in rating_data.keys():
                if(Jaccard_Similarity(key,key1)):
                    score_sen.append(rating_data[key1])
                    break
        
        review_ratings=[int(x[0].strip()[0])/10 for x in score_sen]
        review_confidence=[int(x[1].strip()[0])/5 for x in score_sen]

        assert len(r_sen)==len(review_ratings)==len(review_confidence)


        embed=torch.zeros(8,3,16,768)
        vader_scores=torch.zeros(8,3,16)
        
        des=self.labels[p_id]

        desc=torch.zeros(1)
        
        desc[0]=deci[des]
        r_sen_new={}
        r_sen_senti={}
        
        for i,k in enumerate(r_sen):
            for name,asp in k.items():
               for senti,value in asp.items():
                   if(name not in r_sen_new.keys()):
                       r_sen_new[name]={}

                   if(senti not in r_sen_new[name].keys()):
                       r_sen_new[name][senti]=[]
                #value=[(v,review_ratings[i]*review_confidence[i]) for v in value]
                   r_sen_new[name][senti]+=value
        
        for name,asp in r_sen_new.items():
            for senti,value in asp.items():
            #     value1=[sco[1] for sco in value]
            #     value2=[sco[0] for sco in value]

                value=list(map(preprocess_text,value))
                
                if(senti=='no_sentiment'):
                    value_score=[0]*len(value)
                else:
                    value_score=[get_score(v,senti) for v in value]

                if(len(value_score))<16:
                    value_score=value_score+[0]*(16-len(value_score))
                else:
                    value_score=value_score[0:16]
                
                if(len(value)<16):
                    value=value+[""]*abs(16-len(value))
                    value1=value1+[0]*abs(16-len(value1))
                else:
                    value=value[0:16]
                    value1=value1[0:16]
                    
                input=torch.from_numpy(model.encode(value))
                # scoring_attention=torch.tensor(value1)  #(16)   

                # input=torch.matmul(scoring_attention,input)
                
                value_score=torch.tensor(value_score).squeeze(-1)
                embed[aspect[name]][sentiment[senti]]=input
                vader_scores[aspect[name]][sentiment[senti]]=value_score

        return {"ids":embed,"ids_senti":vader_scores,'targets':desc}

    def __len__(self):
        return self.size


def getLoaders(batch_size,conf_name):
    print('Reading the training Dataset...')
    print()
    train_dataset = Data.getReader(0,8000,conf_name) #19200 #21216
    
    print('Reading the validation Dataset...')
    print()
    valid_dataset = Data.getReader(8000,8800,conf_name) #23200 #25216

    
    trainloader = DataLoader(dataset=train_dataset, batch_size = batch_size, num_workers=0,shuffle=True)
    validloader = DataLoader(dataset=valid_dataset, batch_size = batch_size, num_workers=0,shuffle=True)
   
    return trainloader, validloader

