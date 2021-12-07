import os 
import numpy as np 
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import time
import torchio
#import json
#import cv2
import gc
torch.cuda.empty_cache()
import datetime

from torch.utils.data import Dataset, DataLoader
# import torchvision
#from tqdm.auto import tqdms

import argparse
from src.data.torch_utils import MonkeyEyeballsDataset
from src.models.from_scratch import resnet_for_multimodal_regression as resnet

parser = argparse.ArgumentParser(description='Simulate and show examples of predictions on various monkeys')
parser.add_argument('model_path', type=str,
    help='path to model file')
parser.add_argument('--model_type', type=str, default='resnet50',
    help='type of ResNet - not fully implemented')
parser.add_argument('--labels', default='data/monkey_data.csv', metavar='DF',
    help='path to ICP/IOP dataframe')
parser.add_argument('--scans', default='data/torch_standard', metavar='DIR',
    help='path to dataset folder')
parser.add_argument('--monkey', default=14, type=int, metavar='MONKEY',
    help='which monkey to predict on')
parser.add_argument('--batch', default=4, type=int, metavar='BATCH',
    help='how many images to load into the dataloader at once')
parser.add_argument('--save_path', default=None,
    help='Where to save predictions. Default in same folder as model path.')

def predict(model, dataloader, performances=None):
    if performances is None:
        performances = {}
        performances['pred'] = []
        performances['icp'] = []
        performances['id'] = []
        performances['iop'] = []

    for batch_id, batch_data in enumerate(dataloader):
        icp = batch_data['icp'].float().unsqueeze(1).cuda()
        iop = batch_data['iop'].float().cuda()

        performances['icp'].append(batch_data['icp'].numpy())
        performances['iop'].append(batch_data['iop'].numpy())
        performances['id'].append(batch_data['id'].numpy())

        scan = batch_data['scan'].float().cuda()

        # scan = (scan - 30) / 19
        icp = (icp - 15) / 11
        iop = (iop -22)/ 13

        if device == 'cuda': 
            scan = scan.to(device)
        preds = model(scan.unsqueeze_(1),iop)

        performances['preds'].append(preds.numpy())
        
    return performances
    

def main():
    args = parser.parse_args()
    if args.save_path is None:
        directory = os.path.dirname(args.model_path)
        args.save_path = os.path.join(directory, 'predictions_{}.csv'.format(args.monkey))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file = torch.load(args.model_path)
    model = resnet.resnet50(sample_input_D=128, sample_input_H=128, sample_input_W=512).cuda()
    model.load_state_dict(model_file['state_dict'])

    labels = pd.read_csv(args.labels)
    labels = labels[labels['torch_present'] & ~labels['icp'].isnull() & ~labels['iop'].isnull() & labels['icp'] > 0] 
    labels['icp'] = labels['icp'].astype('float')
    labels['iop'] = labels['iop'].astype('float')
    labels = labels[labels['monkey_id'] == args.monkey]

    med = MonkeyEyeballsDataset(args.scans, labels)
    dataloader = DataLoader(med, batch_size=args.batch, shuffle=False)

    performances = predict(model, dataloader)
    performances = pd.DataFrame(performances)
    performances.to_csv(args.save_path, index=False)
    print('Predictions saved to {}'.format(args.save_path))


if __name__ == '__main__':
    main()