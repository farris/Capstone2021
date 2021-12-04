from torch.utils.data import Dataset, DataLoader
import os 
import numpy as np 
import pandas as pd
import torch
import json


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

# %%
print('-----------------------')

# df = df[df['torch_present']==True]
# df = df[['id','iop','icp']]
# df['id'] = df['id'].apply(np.int64)
# print(df['icp'])
# df['icp'] = df['icp'].apply(np.int64)

# out = MonkeyEyeballsDataset('/scratch/fda239/torch_arrays',df)
# trainloader = torch.utils.data.DataLoader(out, batch_size=2, shuffle=True)
import numpy as np

import pandas as pd


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

print(labels.head())
print(len(labels))


out = MonkeyEyeballsDataset(data_dir,labels)
trainloader = torch.utils.data.DataLoader(out, batch_size=20, shuffle=True)
for batch in trainloader:
    print(batch['iop'].size())
    print(batch['id'].size())
    print(batch['icp'].size())
    print(batch['scan'].size())
    break


#python test2.py
# %%
