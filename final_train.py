# %% Imports
import os
import numpy as np
import pandas as pd
import json
# from src.data.torch_utils import MonkeyEyeballsDataset #NOTE: be aware of path pointer
from torch.utils.data import Dataset, DataLoader
from src.models.MedicalNet.setting import parse_opts
# from src.models.MedicalNet.datasets.brains18 import BrainS18Dataset 
from src.models.MedicalNet.model import generate_model
import torch
import sys
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
from src.models.MedicalNet.utils.logger import log
#from scipy import ndimage
# %%
torch.cuda.empty_cache()
class MonkeyEyeballsDataset(Dataset):
  """
  Loads PyTorch arrays of the monkey eyeballs dataset. 
  """

  def __init__(self, data_dir, labels_df):
    """
    data_dir: path to image scans directory
    labels_df: pandas dataframe holding IOP, ICP and internal Scan ID value
    """
    self.data_dir = data_dir 
    self.labels_df = labels_df

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
   
    scan_path = os.path.join(self.data_dir, '{}.pt'.format(self.labels_df.iloc[idx]['id'].astype('int')))

    sample = {
        'icp':self.labels_df.iloc[idx]['icp'],
        'iop':self.labels_df.iloc[idx]['iop'],
        'scan':torch.load(scan_path),
        'id': self.labels_df.iloc[idx]['id']
    }

    return sample

  def __len__(self):
    return len(self.labels_df)

# %% Load Data
data = pd.read_excel('/scratch/fda239/Monkey Data.xlsx')
data_dir = '/scratch/fda239/torch_arrays'
torch_present = [int(s.strip('.pt')) for s in os.listdir(data_dir)]
keywords = ['Pre', 'pre', 'Norm', 'norm']
labels = data\
  [data['id'].astype(int).isin(torch_present)]\
  [['id', 'iop', 'icp']]
labels['iop'] = np.where(labels['iop'].isin(keywords), '', labels['iop'])
labels['icp'] = np.where(labels['icp'].isin(keywords), '', labels['icp'])
labels = labels.replace(r'^\s*$', np.nan, regex=True)
labels['icp'] = labels['icp'].astype('float')
labels['iop'] = labels['iop'].astype('float')
labels['id'] = labels['id'].astype('int')
labels = labels.dropna()
print(labels)
print('1')
med =  MonkeyEyeballsDataset(data_dir,labels)
print('2')
data_loader = DataLoader(med, batch_size=6, shuffle=True)
print('3')
# for batch in data_loader:
    
#     print(batch['iop'].size())
#     print(batch['id'].size())
#     print(batch['icp'].size())
#     print(batch['scan'].size())
#     break
print('4')

'''
Training code for MRBrainS18 datasets segmentation
Written by Whalechen
'''

# %% Train

def train(data_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets):
    # settings
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    loss_seg = nn.MSELoss()

    print("Current setting is:")
    print(sets)
    print("\n\n")     
    if not sets.no_cuda:
        loss_seg = loss_seg.cuda()
        
    model.train()
    train_time_sp = time.time()
    
    train_losses = []
    batch_id = 1
    print('--------------------------')
    print(total_epochs)
    print(len(data_loader))
    print('--------------------------')
    for epoch in range(total_epochs):
        
        losses = 0
        
        log.info('Start epoch {}'.format(epoch))
        
        scheduler.step()
        log.info('lr = {}'.format(scheduler.get_lr()))
        
        for batch_data in data_loader:
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            label_masks = batch_data['icp']
            volumes = batch_data['scan']

            volumes = (volumes - 30) / 19
            label_masks = (label_masks - 15) / 11 
            
            if not sets.no_cuda: 
                volumes = volumes.cuda()

            optimizer.zero_grad()
        
            #print('-----------------------------', volumes.unsqueeze(1).size())
            print(volumes.unsqueeze(1).size())
            out_masks = model(volumes.unsqueeze(1))#.unsqueeze(1))

            new_label_masks = label_masks.float()
            if not sets.no_cuda:
                new_label_masks = new_label_masks.cuda()
            
            print('prediction', out_masks)

            print('truth', new_label_masks)
            
            # calculating loss
            loss_value_seg = loss_seg(out_masks, new_label_masks)
            loss = loss_value_seg
            loss.backward()
            
            losses += loss.item()
            
            optimizer.step()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                    'Batch: {}-{} ({}), loss = {:.3f}, loss_seg = {:.3f}, avg_batch_time = {:.3f}'\
                    .format(epoch, batch_id, batch_id_sp, loss.item(), loss_value_seg.item(), avg_batch_time))
            
            if not sets.ci_test:
                # save model
                if batch_id == 0 and batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                #if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
                    model_save_dir = os.path.dirname(model_save_path)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)
                    
                    log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id)) 
                    torch.save({
                                'ecpoch': epoch,
                                'batch_id': batch_id,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()},
                                model_save_path)
            batch_id += 1                        
        train_losses.append(losses)

        print(train_losses)
        
    print('Finished training')  
    
    if sets.ci_test:
        exit()

# %% Setting

sets = parse_opts()   
if sets.ci_test:
    sets.img_list = './toy_data/test_ci.txt' 
    sets.n_epochs = 1
    sets.no_cuda = True
    sets.data_root = './toy_data'
    sets.pretrain_path = ''
    sets.num_workers = 0
    sets.model_depth = 10
    sets.resnet_shortcut = 'A'
    sets.input_D = 14
    sets.input_H = 28
    sets.input_W = 28
    
# getting model ####################
torch.manual_seed(sets.manual_seed)
sets.gpu_id = [0]
sets.pretrain_path = 'src/models/MedicalNet/pretrain/resnet_50.pth'
model, parameters = generate_model(sets) 

# optimizer ####################
if sets.ci_test:
    params = [{'params': parameters, 'lr': sets.learning_rate}]
else:
    params = [
            { 'params': parameters['base_parameters'], 'lr': sets.learning_rate }, 
            { 'params': parameters['new_parameters'], 'lr': sets.learning_rate }
            ]

optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)   
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# train from resume ####################
if sets.resume_path:
    if os.path.isfile(sets.resume_path):
        print("=> loading checkpoint '{}'".format(sets.resume_path))
        checkpoint = torch.load(sets.resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
          .format(sets.resume_path, checkpoint['epoch']))

# getting data ####################
sets.phase = 'train'
if sets.no_cuda:
    sets.pin_memory = False
else:
    sets.pin_memory = True    

# training ####################
train(data_loader, model, optimizer, scheduler, total_epochs=15 ,save_interval=sets.save_intervals, save_folder=sets.save_folder, sets=sets)