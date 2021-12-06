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

  def __init__(self, data_dir, labels_df, transform=None):
    """
    data_dir: path to image scans directory
    labels_df: pandas dataframe holding IOP, ICP and internal Scan ID value
    """
    self.data_dir = data_dir 
    self.labels_df = labels_df
    self.transform = transform

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
    if self.transform:
        sample['scan'] = self.transform(sample['scan'].unsqueeze(0)).squeeze()

    return sample

  def __len__(self):
    return len(self.labels_df)