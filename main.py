

import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from   model import AADP
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.autonotebook import tqdm


import os
import torch
import matplotlib.pyplot as plt
from dataloader import getLoaders
from tqdm.autonotebook import tqdm
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,classification_report

batch_size=16

torch.manual_seed(400)
np.random.seed(400)

train_loader,val_loader=getLoaders(batch_size,"")
os.environ['TOKENIZERS_PARALLELISM']='False'

print("Length of TrainLoader:",len(train_loader))
print("Length of ValidLoader:",len(val_loader))


if torch.cuda.is_available():      
    device = torch.device("cuda:1")
else:
  print('No GPU available, using the CPU instead.')
  device = torch.device("cpu")

text_model=AADP()
text_model.to(device)
criterion = nn.BCELoss()

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in text_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in text_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.01}
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=0.001, weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')

text_model.train()
result=[]
EPOCH= 12
print("number of epoch = ",EPOCH)

loss_log1 = []
loss_log2 = []

train_f1_log=[]

val_f1_log=[]

train_acc_log=[]
val_acc_log=[]

final_output_train=[]
final_output_val=[]

best_val_out=[]
max_acc=0

for epoch in range(EPOCH):
  
  train_out = []
  train_true = []

  val_out = []
  val_true = []

  final_train_loss=0.0
  final_val_loss=0.0

  l1 = []
  l2 = []

  text_model.train()

  for data in tqdm(train_loader,desc="Train epoch {}/{}".format(epoch + 1, EPOCH)):

    ids = data['ids'].to(device)
    ids_senti = data['ids_senti'].to(device)
    targets = data['targets'].to(device,dtype = torch.float)

    t1 = (ids,ids_senti)
    
    optimizer.zero_grad()
    out,w_m= text_model(t1)

    loss =criterion(out, targets)
    
    train_out+=out.squeeze().tolist()
    train_true+=targets.squeeze().tolist()
    
    l1.append(loss.item())
    final_train_loss +=loss.item()
    loss.backward()
    optimizer.step()

  loss_log1.append(np.average(l1))

  with torch.no_grad():
    text_model.eval()

    for data in tqdm(val_loader,desc="Valid epoch {}/{}".format(epoch + 1, EPOCH)):
      ids = data['ids'].to(device)
      ids_senti = data['ids_senti'].to(device)
      targets = data['targets'].to(device,dtype = torch.float)
      
      t1 = (ids,ids_senti)
      
      out_val,_= text_model(t1)
      
      val_out+=out_val.squeeze().tolist()
      val_true+=targets.squeeze().tolist()
    
      loss = criterion(out_val, targets)
      l2.append(loss.item())
      
      final_val_loss+=loss.item()

    #scheduler.step(final_val_loss)
    loss_log2.append(np.average(l2))
    curr_lr = optimizer.param_groups[0]['lr']
  
  train_out=np.array([s>0.5 for s in train_out])
  train_true=np.array(train_true)

  val_out=np.array([s>0.5 for s in val_out])
  val_true=np.array(val_true)


  print("Epoch {}, loss: {}, val_loss: {}".format(epoch+1, final_train_loss/len(train_loader) ,final_val_loss/len(val_loader)))
  print(f"TRAINING F1 SCORE: {f1_score(train_true,train_out,average='weighted')} \nValidation F1 SCORE: {f1_score(val_true,val_out,average='weighted')}")
  print(f"TRAINING ACCURACY: {accuracy_score(train_true,train_out)} \nValidation ACCURACY: {accuracy_score(val_true,val_out)}")
  w_m=w_m.detach().cpu()
  #s_i=s_i.detach().cpu()

  if(epoch==EPOCH-1):
    print(classification_report(val_true,val_out))
    with open("weight_matrix_JCDL.pickle",'wb') as out:
      pickle.dump(w_m,out)


  train_f1_log.append(f1_score(train_true,train_out))
  val_f1_log.append(f1_score(val_true,val_out))

  train_acc_log.append(accuracy_score(train_true,train_out))
  val_acc_log.append(accuracy_score(val_true,val_out))

  
  if(accuracy_score(val_true,val_out)>max_acc):
    torch.save(text_model.state_dict(), "final_model.pt")
    max_acc=accuracy_score(val_true,val_out)
    best_val_out=val_out

    
plt.plot(range(len(loss_log1)), loss_log1)
plt.plot(range(len(loss_log2)), loss_log2)
plt.savefig('loss_curve.png')
print("MAXIMUM ACCURACY:",max_acc)
