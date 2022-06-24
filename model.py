
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AutoModel
import numpy as np

torch.manual_seed(400)
np.random.seed(400)


os.environ['TOKENIZERS_PARALLELISM']='False'

#If there's a GPU available...
if torch.cuda.is_available():    

    device = torch.device("cuda:1")
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
  print('No GPU available, using the CPU instead.')
  device = torch.device("cpu")
  

class AADP(nn.Module):
    def __init__(self):

        super(AADP, self).__init__()
        
        self.flatten=nn.Flatten()
        
        self.conv1 = nn.Conv3d(8, 8, kernel_size=(1, 9, 1))
        self.conv2 = nn.Conv3d(8, 8, kernel_size=(1, 5, 1))
        self.conv3 = nn.Conv3d(8, 8, kernel_size=(1, 4, 1))
        
        self.linear1=     nn.Linear(768,512)
        self.linear2=     nn.Linear(512,256)
        self.linear3=     nn.Linear(256,128)
         
        self.linear4=     nn.Linear(144,128)
        self.linear5=     nn.Linear(128,16)
        self.last_dense = nn.Linear(16,1)

        self.dropout1=nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.sigmoid=nn.Sigmoid()

        imp_score= torch.rand(8, 1,requires_grad=True)  #(512,1)
        nn.init.xavier_normal_(imp_score)
        
        imp_score_senti= torch.rand(3, 1,requires_grad=True)  #(512,1)
        nn.init.xavier_normal_(imp_score_senti)

        self.imp_score=imp_score
        self.imp_score_senti=imp_score_senti
        
        
    def forward(self, t1):
        ids=t1[0]                                                                      
        ids = self.conv1(ids)                                                     
        #print(ids.size())
        ids = self.conv2(ids)                                                    
        #print(ids.size())
        ids = self.conv3(ids)                                                     
        #print(ids.size())

        ids = ids.squeeze(3)                                                      
        
        v_senti=t1[1]                                                             
        
        l = self.relu(self.linear1(ids))                                         
        l = self.dropout1(l)                              
        l = self.relu(self.linear2(l))                                            
        l = self.dropout1(l)   
        l = self.relu(self.linear3(l))                                           

        aspect_senti=torch.cat((l,v_senti),dim=3)                                 
        aspect_senti=aspect_senti.permute(0,1,3,2)                               
        i_p_senti = F.softmax(self.imp_score_senti,dim=0).to(device)
        i_p_senti = i_p_senti.unsqueeze(0).unsqueeze(0)                           
        
        aspect_senti=torch.matmul(aspect_senti,i_p_senti).squeeze(-1).to(device)   
        aspect_senti=aspect_senti.permute(0,2,1)                                    
        
        i_p = F.softmax(self.imp_score,dim=0).to(device)
        i_p = i_p.unsqueeze(0)                                                     

        e=torch.matmul(aspect_senti,i_p).squeeze(-1).to(device)                    
        
        l = self.relu(self.linear4(e))                                                                       
        l = self.relu(self.linear5(l))                                              
        model_output = self.sigmoid(self.last_dense(l))                             
        
        del l
        return model_output,i_p
