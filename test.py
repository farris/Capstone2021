# %%
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
# %%
class MyDataset(Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        assert x.shape[0] == y.shape[0] # assuming shape[0] = dataset size
        self.x = x
        self.y = y


    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

# %%
class CNN(torch.nn.Module):

  def __init__(self):
    
    super(CNN, self).__init__()
    
    ######################################################
    #inputs###############################################
    self.c1_out = 2                                     ##
                                                        ##
    self.c2_out = 15                                    ##
                                                        ##
    self.c3_out = 12                                    ##
                                                        ##
    self.c4_out = 5                                     ##
    ######################################################


    #1st layer ###########################################
    self.conv1 = nn.Conv3d(in_channels = 1, out_channels = self.c1_out,\
                           kernel_size = (5,5,5), stride=(30,30,30), padding=0) ## frames 
    #relu

    
    ######################################################


    #Linear layer ########################################
    self.fc1 = nn.Linear(850, 1)
    #relu
    ######################################################

  def forward(self, x):
    out = F.relu(self.conv1(x))
    print('-----')
    print(out.shape)
    out = out.view(-1, out.size()[1] * out.size()[2] * out.size()[3] * out.size()[4])
    out = F.relu(self.fc1(out))
    print('-----')
    print(out.shape)
    return out
# %%
def get_loss_and_correct(model, batch, criterion, device):
  # Implement forward pass and loss calculation for one batch.
  # Remember to move the batch to device.
  # 
  # Return a tuple:
  # - loss for the batch (Tensor)
  # - number of correctly classified examples in the batch (Tensor)

  imgs = batch[0].cuda()

  labels = torch.tensor(batch[1]).unsqueeze_(1)
  labels = labels.type(torch.float32).cuda()
    

  outputs = model(imgs)
  print(outputs)
  print(labels)
  print(outputs.size())
  print(labels.size())

  loss = criterion(outputs,labels)
  print(loss)
  preds = outputs.data.max(1, keepdim=True)[1]

  return loss,torch.eq(preds.flatten(),labels).sum()


def step(loss, optimizer):
  # Implement backward pass and update.
  # TODO
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()


# %%

print(torch.cuda.is_available())
print('--------------------------------')
im = torch.load('/scratch/fda239/1064.pt')
im = im.unsqueeze_(0)


data = [im for i in range(6)]
data_torch = torch.stack(data)
ICP = [30,40,50,60,30,40]

traindata = MyDataset(data_torch, torch.tensor(ICP))
trainloader = torch.utils.data.DataLoader(traindata, batch_size=2, shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


total_train_loss = 0.0
total_train_correct = 0.0
for batch in trainloader:
    print('start---------------------------')
    loss, correct = get_loss_and_correct(model, batch, criterion, device = None)
    step(loss, optimizer)
    total_train_loss += loss.item()
    total_train_correct += correct.item()
    print('end---------------------------')

# %%


