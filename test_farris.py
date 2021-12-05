import os 
import numpy as np 
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import time
import torch.nn.functional as F
#import json
#import cv2

def R2Loss(y_pred,y):
    print(y)
    var_y = torch.var(y)
    print(var_y)
    print(F.mse_loss(y_pred, y, reduction="mean"))
    return 1.0 - F.mse_loss(y_pred, y, reduction="mean") / var_y




from torch.utils.data import Dataset, DataLoader
# import torchvision
#from tqdm.auto import tqdms


seed = 12345
torch.manual_seed(seed)

from src_eric.data.torch_utils import MonkeyEyeballsDataset
from src_eric.models.from_scratch import resnet_for_multimodal_regression as resnet




labels = pd.read_excel('/scratch/fda239/Monkey Data.xlsx')
labels = labels[labels['torch_present'] & ~labels['icp'].isnull() & ~labels['iop'].isnull() & labels['icp'] > 0] 
labels['icp'] = labels['icp'].astype('float')
labels['iop'] = labels['iop'].astype('float')

# print(labels)
# train_labels = labels[labels['monkey_id'] != 14]
# # 8 handpicked examples 
# val_examples = [1751, 1754, 1761, 1766]
# val_labels = labels[labels['id'].isin(val_examples)]

train_labels =labels.sample(frac=0.90,random_state=200) #random state is a seed value
val_labels =labels.drop(train_labels.index)


print(len(train_labels))
print(len(val_labels))


med_train = MonkeyEyeballsDataset('/scratch/fda239/torch_arrays', train_labels)
med_val = MonkeyEyeballsDataset('/scratch/fda239/torch_arrays', val_labels)


dataloader_train = DataLoader(med_train, batch_size=6,shuffle=True) 
dataloader_val = DataLoader(med_val, batch_size=5, shuffle=False)

print(len(dataloader_train))
print(len(dataloader_val))


model = resnet.resnet10(sample_input_D=128, sample_input_H=128, sample_input_W=512)
OPTIMIZER = torch.optim.Adamax(model.parameters(), lr=0.0001)
warm_start = torch.load('/scratch/fda239/12_3_run/epoch_19_batch_87.pth.tar',map_location=torch.device('cpu')) 
model.load_state_dict(warm_start['state_dict'])
OPTIMIZER.load_state_dict(warm_start['optimizer'])

temp = []

loss = nn.MSELoss(reduction='mean')

with torch.no_grad():
    for batch_data_val in dataloader_val:
        icp_val = batch_data_val['icp'].float().unsqueeze(1)
        iop_val = batch_data_val['iop'].float()
        scan_val = batch_data_val['scan'].float()
        
        scan_val = (scan_val - 30) / 19
        icp_val = (icp_val - 15) / 11

    
        preds_val = model(scan_val.unsqueeze_(1),iop_val)
        loss_value_val = loss(preds_val, icp_val)
        print(R2Loss(preds_val,icp_val))
        temp.append(loss_value_val.item())

print(temp)
print(np.mean(temp))
