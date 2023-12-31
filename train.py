# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.optim
import os
import torch.utils.model_zoo as model_zoo
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset


def fit(net, optimizer, lr_scheduler, loss_func, n_epochs, train, val):    
    net.cuda()
    loss_train = []
    avg_rate_test = []
    loss_val = []        
    for e in range(n_epochs):
        print(e)
        net.train() #put the net in train mode
        loss = []
        for x, y in train:
            x = x.cuda()
            y = y.cuda()
            # x = F.interpolate(x, scale_factor=7, mode='bilinear', align_corners=True)
            loss.append(loss_batch(net, loss_func, x, y, optimizer)) 
        loss_train.append(sum(loss)/len(loss))
        print('train loss is ' + str(sum(loss)/len(loss)))
        lr_scheduler.step()
        loss = []
        with torch.no_grad():
            net.eval() #put the net in evaluation mod
            labels_predict=np.array([])
            labels_true=np.array([])
            for xb, yb in val:
                xb = xb.cuda()
                yb = yb.cuda()
                loss.append(loss_batch(net, loss_func, xb, yb)) 
                # xb = F.interpolate(xb, scale_factor=7, mode='bilinear', align_corners=True)

                yb_pre = net(xb)
                yb_pre = yb_pre.detach()
                yb_pre = torch.argmax(yb_pre, dim = 1)
                yb = yb.detach().cpu().numpy()
                yb_pre = yb_pre.cpu().numpy()
                labels_true = np.append(labels_true,yb.reshape(-1))
                labels_predict = np.append(labels_predict,yb_pre.reshape(-1))

            loss_val.append(sum(loss)/len(loss))
            print('val loss is ' + str(sum(loss)/len(loss)))
            C = confusion_matrix(labels_true, labels_predict)
            print(C)
            avg = accuracy(labels_predict, labels_true)
            print('test accuracy is ' + str(avg))
            print("--------------------------------------------------")
            
            avg_rate_test.append(avg)
        torch.save(net.state_dict(), str(e)+"_"+str(avg)+".pb")

    return loss_train, avg_rate_test

def loss_batch(model, loss_func, xb, yb, opt=None):
    """ Compute the loss of the model on a batch of data, or do a step of optimization.

    @param model: the neural network
    @param loss_func: the loss function (can be applied to model(xb), yb)
    @param xb: a batch of the training data to input to the model
    @param yb: a batch of the training labels to input to the model
    @param opt: a torch.optimizer.Optimizer.  If not None, use the Optimizer to improve the model. Otherwise, just compute the loss.
    @return a numpy array of the loss of the minibatch, and the length of the minibatch
    """
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        for group in opt.param_groups:
            for p in group['params']:
                state = opt.state[p]
                if 'step' in state.keys():
                    if(state['step']>=1024):
                        state['step'] = 1000
        opt.step()
        opt.zero_grad()

    return loss.item()

def accuracy(y_pre, y):
    avg_class_rate = 0.0    
    n = y.shape[0]
    
    for i in range(n):
        if y_pre[i] == y[i]:
            avg_class_rate += 1.0
    avg_class_rate = avg_class_rate/n
    
    return avg_class_rate

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def resnet18(pretrained=True):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    # model = torchvision.models.resnet18()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'],model_dir='./'))
    return model


class MyDataset(Dataset):
    """ My dataset."""

    # Initialize your data, download, etc.
    def __init__(self,case=None,transform=None):
        dir_path = "datasets/"
        if(case=='train'):
            print("Train dataset loading")

            File1 = dir_path+'count_matrix_train_noLPS_filter2.csv'
            xy = np.loadtxt(File1,delimiter=',', dtype=np.float32)
            xy = xy.T
            #print(xy.shape)
            File1 = dir_path+'label_train_4_noLPS2.txt'
            xy_label = np.loadtxt(File1,delimiter='\t', dtype=np.int64, encoding='utf-16')
            #print(xy_label.shape)
            self.transform=transform
            self.x_data=xy
            self.len=self.x_data.shape[0]
            self.y_data = xy_label

        if(case=='test'):
            print("Test dataset loading")
            File = dir_path+'count_matrix_val_noLPS_filter2.csv'
            xy = np.loadtxt(File,delimiter=',', dtype=np.float32)
            xy = xy.T
            #print(xy.shape)
            File = dir_path+'label_val_4_noLPS2.txt'
            xy_label = np.loadtxt(File,delimiter='\t', dtype=np.int64, encoding='utf-16')
            self.transform=transform
            self.x_data=xy
            self.len=self.x_data.shape[0]
            self.y_data = xy_label


    def __getitem__(self, index):
        return self.transform(self.x_data[index]),self.y_data[index]

    def __len__(self):
        return self.len



class MyDataset1(Dataset):
    """ My dataset."""

    # Initialize your data, download, etc.
    def __init__(self,case=None,transform=None):
        dir_path = "datasets/"
        if(case=='train'):
            print("Train dataset loading")

            File1 = dir_path+'count_matrix_train_noLPS_filter1.csv'
            xy = np.loadtxt(File1,delimiter=',', dtype=np.float32)
            xy = xy.T
            print(xy.shape)
            File1 = dir_path+'label_train_4_noLPS1.txt'
            xy_label = np.loadtxt(File1,delimiter='\t', dtype=np.int64, encoding='utf-16')
            print(xy_label.shape)
            self.transform=transform
            self.x_data=xy
            self.len=self.x_data.shape[0]
            self.y_data = xy_label

        if(case=='test'):
            print("Test dataset loading")
            File = dir_path+'count_matrix_val_noLPS_filter1.csv'
            xy = np.loadtxt(File,delimiter=',', dtype=np.float32)
            xy = xy.T
            print(xy.shape)
            File = dir_path+'label_val_4_noLPS1.txt'
            xy_label = np.loadtxt(File,delimiter='\t', dtype=np.int64, encoding='utf-16')
            self.transform=transform
            self.x_data=xy
            self.len=self.x_data.shape[0]
            self.y_data = xy_label
    def __getitem__(self, index):
        #print("Index",index)
        return self.transform(self.x_data[index]),self.y_data[index]

    def __len__(self):
        return self.len
            
class ArrayToTensor_train(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to
        a list of torch.FloatTensor of shape (C x H x W)."""
    def __call__(self, gene_exp):
        a = 221 
        b = 222
        gene_exp_new = np.reshape(gene_exp, (1, a, b))
        # adding Guassian noise to each data
        noise_factor = 25
        noise_data = np.random.randn(1, gene_exp_new.shape[1],gene_exp_new.shape[2])*noise_factor
        gene_exp_new = gene_exp_new + noise_data
        gene_exp_new = np.where(gene_exp_new < 0, 0, gene_exp_new)

        gene_exp_new=torch.from_numpy(gene_exp_new).float()
        return gene_exp_new

class ArrayToTensor_test(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to
        a list of torch.FloatTensor of shape (C x H x W)."""
    def __call__(self, gene_exp):
        a = 221# 221
        b = 222# 222
        gene_exp_new = np.reshape(gene_exp, (1, a, b))
        gene_exp_new=torch.from_numpy(gene_exp_new).float()
        return gene_exp_new

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    batch_size = 32
    # # For trainning data
    trainset = MyDataset(case='train',transform=ArrayToTensor_test())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8) 
    # For testing data 
    testset = MyDataset(case='test',transform=ArrayToTensor_test())
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    print("done")    
    
    net = resnet18(pretrained=True)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.fc = nn.Linear(net.fc.in_features, 4)
    net.cuda()
    
    weight_CE = torch.FloatTensor([1, 3, 2, 2]).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=weight_CE)
    opt = torch.optim.Adam(net.parameters(),lr=0.000005, weight_decay=0.0005, amsgrad=True)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)
    
    
    train_el, val_el = fit(net, opt, sch, loss_fn, 50, trainloader, testloader)
    print(train_el,val_el)
