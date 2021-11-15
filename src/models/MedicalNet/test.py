%%writefile  MedicalNet/test.py
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
from sklearn.metrics import accuracy_score
from skimage import io


def test(data_loader, model, img_names, sets):
    masks = []
    model.eval() # for testing 
    for batch_id, batch_data in enumerate(data_loader):
        # forward
        volume = batch_data
        if not sets.no_cuda:
            volume = volume.cuda()
        with torch.no_grad():
            probs = model(volume)
            probs = F.softmax(probs, dim=1)

        # resize mask to original size
        [batchsize, _, mask_d, mask_h, mask_w] = probs.shape
        data = io.imread(os.path.join(sets.data_root, img_names[batch_id]))
        [depth, height, width] = data.shape
        mask = probs[0]
        scale = [1, depth*1.0/mask_d, height*1.0/mask_h, width*1.0/mask_w]
        mask = ndimage.interpolation.zoom(mask.cpu(), scale, order=1)
        mask = np.argmax(mask, axis=0)   
        masks.append(mask)
 
    return masks

if __name__ == '__main__':
    # settting
    sets = parse_opts()
    sets.target_type = "normal"
    sets.phase = 'test'

    # getting model
    checkpoint = torch.load(sets.resume_path)
    net, _ = generate_model(sets)
    net.load_state_dict(checkpoint['state_dict'])

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
    for idx in range(Nimg):
        label = np.load(os.path.join(sets.data_root, label_names[idx]))
        label = label.f.arr_0
        print('masks[idx]', np.max(masks[idx]), np.min(masks[idx]))
        accuracies[idx, :] = accuracy_score(masks[idx].flatten(), label.flatten())
    
    # print result
    for idx in range(1, sets.n_seg_classes):
        print(accuracies)
        mean_accuracy_per_task = np.mean(accuracies[idx, :])
        print('mean accuracy for class-{} is {}'.format(idx, mean_accuracy_per_task))