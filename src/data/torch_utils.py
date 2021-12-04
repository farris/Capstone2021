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

  def init(self, data_dir, labels_df):
    """
    data_dir: path to image scans directory
    labels_df: pandas dataframe holding IOP, ICP and internal Scan ID value
    """
    self.data_dir = data_dir 
    self.labels_df = labels_df

  def getitem(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    scan_path = os.path.join(self.data_dir, '{}.pt'.format(self.labels_df.iloc[idx]['id']))

    sample = {
        'icp':self.labels_df.iloc[idx]['icp'],
        'iop':self.labels_df.iloc[idx]['iop'],
        'scan':torch.load(scan_path),
        'id': self.labels_df.iloc[idx]['id']
    }

    return sample

  def len(self):
    return len(self.labels_df)
