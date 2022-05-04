

import random
import pickle
import os
import jsonlines
import nltk
import json
nltk.download('punkt',quiet=True)
from tqdm.autonotebook import tqdm


if(os.path.isdir('input_files')!=True):
    os.mkdir('input_files')


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
 

def create_data(conf_name):
    data_path=r'raw_dataset/dataset/aspect_data'
    r_data={}
    counter=0

    phrase_data={}
    check_data={}

    with jsonlines.open(os.path.join(data_path,'review_with_aspect.jsonl')) as f:
        pbar=f.iter()
        
        for line in tqdm(pbar,desc='Loading Phrases from Data'):
            id1=line['id']
            if(not id1.startswith(conf_name)):
                continue
            s=line['text']
            labels=line['labels']
            
            if (id1 not in phrase_data.keys()):
                phrase_data[id1]={}
                check_data[id1]={}
                
            aspect={}
            for k in labels:
                
                if(k[2].startswith('summary')):
                    continue
                    
                a=k[2].split('_')
                
                if(a[0]=='meaningful'):
                    a=(a[0]+'_'+a[1],a[2])
                else:
                    a=(a[0],a[1])

                h=s[k[0]:k[1]]
                
                
                if(a[0] in aspect):
                    
                    if(a[1] not in aspect[a[0]]):
                        aspect[a[0]][a[1]]=[]
                        aspect[a[0]][a[1]].append(h)
                        
                    else:
                        aspect[a[0]][a[1]].append(h)
                            
                else:
                    aspect[a[0]]={}
                    aspect[a[0]][a[1]]=[]
                    aspect[a[0]][a[1]].append(h)
                    
            
            phrase_data[id1][s]=aspect
            check_data[id1][s]=labels
            counter+=1

            
    data_path=r'raw_dataset/dataset/'
    decision_data={}
    rating_data={}

    for conf in os.listdir(data_path):
        if(not conf.startswith(conf_name)):
            continue
            
        for dire in (os.listdir(os.path.join(data_path,conf))):
            
            if(dire.endswith('_content')):
                continue
            if(dire.endswith('_review')):
                for paper in tqdm(os.listdir(os.path.join(data_path,conf,dire)),desc=conf+" RATINGS "+": done"):
                    
                    with open(os.path.join(data_path,conf,dire,paper)) as out:
                        file1=json.load(out)
                    
                    id=file1['id']

                    if(id not in rating_data.keys()):
                        rating_data[id]={}
                    
                    for rev in file1['reviews']:
                        if('rating' not in rev.keys()):
                            rev['rating']="5:XX"
                        if('confidence' not in rev.keys()):
                            rev['confidence']="2:XX "

                        rating_data[id][rev['review']]=(rev['rating'],rev['confidence'])
                     

            if(dire.endswith('_paper')): 
                for paper in tqdm(os.listdir(os.path.join(data_path,conf,dire)),desc=conf+" DECISIONS "+": done"):

                    with open(os.path.join(data_path,conf,dire,paper)) as out:
                        file1=json.load(out)

                    decision=file1['decision']
                    decision_data[file1['id']]='accept' if ('Accept' in decision or 'Track' in decision) else 'reject'
        print()
                    
    print("Total number of papers    :",len(decision_data))
    print("Number of Accepted Papers :",list(decision_data.values()).count('accept'))
    print("Number of Rejected Papers :",list(decision_data.values()).count('reject'))

    s1=list(zip(list(phrase_data.keys()),list(phrase_data.values())))

    random.shuffle(s1)
    a,b = zip(*s1)

    p_data={k:v for k,v in zip(a,b)}

    with open("input_files/paper_review_phrases_JCDL"+conf_name+".pickle",'wb') as out:
        pickle.dump(p_data,out)
        
    with open("input_files/paper_decision_JCDL"+conf_name+".pickle",'wb') as out:
        pickle.dump(decision_data,out)
     
    with open("input_files/paper_rating_JCDL"+conf_name+".pickle",'wb') as out:
        pickle.dump(rating_data,out)


    # count=0
    # for id in phrase_data.keys():
    #     for s in phrase_data[id].keys():
    #         flag=0
    #         for key1 in rating_data[id].keys():
    #             if(Jaccard_Similarity(s,key1)):
    #                 flag=1
    #                 break

    #         if(flag==0):
    #             count+=1
    
    # print(count)


        