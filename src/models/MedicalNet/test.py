from setting import parse_opts 
from datasets.brains18 import BrainS18Dataset
from model import generate_model
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy import ndimage
import nibabel as nib
import sys
import os
from utils.file_process import load_lines
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score
from skimage import io

def test(data_loader, model, img_names, sets):
    masks = []
    model.eval() 
    
    # for testing 
    for batch_id, batch_data in enumerate(data_loader):
        # forward
        volume = batch_data
        if not sets.no_cuda:
            volume = volume.cuda()
        with torch.no_grad():
            predictions = model(volume)

        mask = predictions
        masks.append(mask.float())
 
    return masks

if __name__ == '__main__':
    # setting
    sets = parse_opts()
    sets.target_type = "normal"
    sets.phase = 'test'

    # getting model
    #checkpoint = torch.load(sets.resume_path)
    net, _ = generate_model(sets)
    #net.load_state_dict(checkpoint['state_dict'])

    # data tensor
    testing_data = BrainS18Dataset(sets.data_root, sets.img_list, sets)
    data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    # testing
    img_names = [info.split(" ")[0] for info in load_lines(sets.img_list)]
    masks = test(data_loader, net, img_names, sets)
    
    # evaluation: calculate accuracy 
    label_names = [info.split(" ")[1] for info in load_lines(sets.img_list)]
    Nimg = len(label_names)
    accuracies = np.zeros([Nimg, sets.n_seg_classes])
    predictions = masks
    labels = np.array([])
    for idx in range(Nimg):
        label = np.load(os.path.join(sets.data_root, label_names[idx]))
        label = np.mean(label.f.arr_0)
        labels = np.append(labels, label)
        
    print('actual y:', labels)
    print('predicted y:', masks)
    
    r_squared = r2_score(labels, masks)
    
    # print result
    print('r squared is {}'.format(r_squared))