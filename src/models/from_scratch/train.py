import os 
import numpy as np 
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import time
import json
import cv2

import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
import torchvision
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 12345
torch.manual_seed(seed)

from src.data.torch_utils import MonkeyEyeballsDataset
from src.models.from_scratch import resnet_for_multimodal_regression as resnet

def train(dataloader_train, 
          dataloader_val, 
          model, 
          optimizer, 
          scheduler, 
          val_interval,
          save_interval, 
          save_folder,
          warm_start_epoch=0,
          loss=nn.MSELoss(reduction='sum'), 
          total_epochs=100):
    # settings
    batches_per_epoch = len(dataloader_train)
    print('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))

    if device == 'cuda':
      loss = loss.to(device)
        
    model.train()
    train_time_sp = time.time()

    for epoch in range(warm_start_epoch, total_epochs):
        print('Start epoch {}'.format(epoch))
        
        for batch_id, batch_data in enumerate(dataloader_train):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch + batch_id
            icp = batch_data['icp'].float().unsqueeze(1)
            iop = batch_data['iop'].float()
            scan = batch_data['scan'].float()

            if device == 'cuda': 
                scan = scan.to(device)

            # standardize input
            scan = (scan - 30) / 19
            icp = (icp - 15) / 11 

            optimizer.zero_grad()
            # add fake channel dimension as 5-D input is expected
            preds = model(scan.unsqueeze(1))
            
            # calculating loss
            loss_value = loss(preds, icp)
            loss_value.backward()                
            optimizer.step()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            print(
                'Batch: {}-{} ({}), loss = {:.3f}, avg_batch_time = {:.3f}'\
                .format(epoch, batch_id, batch_id_sp, loss_value, avg_batch_time))
          
            # get validation loss
            if batch_id_sp % val_interval == 0:
                model.eval()
                print('')
                print('Validating...')
                for batch_id_val, batch_data_val in enumerate(dataloader_val):
                    icp_val = batch_data_val['icp'].float().unsqueeze(1)
                    iop_val = batch_data_val['iop'].float()
                    

                    scan_val = batch_data_val['scan'].float()
                    scan_val = (scan_val - 30) / 19
                    icp_val = (icp_val - 15) / 11

                    if device == 'cuda': 
                        scan_val = scan_val.to(device)
                    preds_val = model(scan_val)
                    loss_value_val = loss(preds_val, icp_val)
                    
                    print('Loss on validation: {:.3f}'.format(loss_value_val))
                    print('')
                
                model.train()

            # save model
            if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                model_save_path = os.path.join(save_folder, 'epoch_{}_batch_{}.pth.tar'\
                                               .format(epoch, batch_id))
                model_save_dir = os.path.dirname(model_save_path)
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                
                print('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id)) 
                torch.save({
                            'epoch': epoch,
                            'batch_id': batch_id,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                            model_save_path)
        scheduler.step()
        print('lr = {}'.format(scheduler.get_lr()))
                           
    print('Finished training')  

labels = pd.read_csv('data/monkey_data.csv')
labels = labels[labels['torch_present'] & ~labels['icp'].isnull() & ~labels['iop'].isnull() & labels['icp'] > 0] 
labels['icp'] = labels['icp'].astype('float')
labels['iop'] = labels['iop'].astype('float')
train_labels = labels[labels['monkey_id'] != 14]
# 8 handpicked examples 
val_examples = [1751, 1754, 1761, 1766]
val_labels = labels[labels['id'].isin(val_examples)]

med_train = MonkeyEyeballsDataset('data/torch_arrays_128', train_labels)
med_val = MonkeyEyeballsDataset('data/torch_arrays_128', val_labels)

dataloader_train = DataLoader(med_train, batch_size=8, num_workers=2, shuffle=True, pin_memory=True) 
dataloader_val = DataLoader(med_val, batch_size=4, num_workers=2, shuffle=False, pin_memory=True)

model = resnet.resnet10(sample_input_D=128, sample_input_H=128, sample_input_W=512)
EPOCHS = 100
OPTIMIZER = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-3)
SCHEDULER = lr_scheduler.ExponentialLR(OPTIMIZER, gamma=0.99)
LOSS = nn.MSELoss(reduction='sum')

# load in in case of warm start
warm_start = torch.load('models/models/epoch_0_batch_100.pth.tar')
model.load_state_dict(warm_start['state_dict'])
OPTIMIZER.load_state_dict(warm_start['optimizer'])

if warm_start.get('epoch') is not None:
    current_epoch = warm_start.get('epoch')
else:
    current_epoch = 0

train(dataloader_train=dataloader_train, 
      dataloader_val=dataloader_val,
      model=model, 
      optimizer=OPTIMIZER, 
      scheduler=SCHEDULER, 
      total_epochs=EPOCHS, 
      warm_start_epoch=current_epoch,
      save_interval=25, 
      save_folder='models/models/run_11_28_2021', # change this for a new run or change to pass it in as command line arg
      val_interval=10,
      loss=LOSS)
