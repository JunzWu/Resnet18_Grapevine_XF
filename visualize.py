# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import os
import torch.utils.model_zoo as model_zoo
import visualisation.core.SaliencyMap as SaliencyMap

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def resnet18(pretrained=True):
    # model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    model = torchvision.models.resnet18()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'],model_dir='./'))
    return model

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    net = resnet18(pretrained=True)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.fc = nn.Linear(net.fc.in_features, 4)
    net.load_state_dict(torch.load("/home/junz/Desktop/Internship/54_0.7232142857142857.pb"))
    net.eval() 


    print("Test dataset loading")
    dir_path = "datasets/"
    File1 = dir_path+'count_matrix_train_noLPS_filter1.csv'
    xy1 = np.loadtxt(File1,delimiter=',', dtype=np.float32)
    xy1 = xy1.T
    File2 = dir_path+'count_matrix_val_noLPS_filter1.csv'
    xy2 = np.loadtxt(File2,delimiter=',', dtype=np.float32)
    xy2 = xy2.T
    xy = np.concatenate((xy1,xy2),axis=0)


    File1 = dir_path+'label_train_4_noLPS.txt'
    xy_label1 = np.loadtxt(File1,delimiter='\t', dtype=np.int64, encoding='utf-16')
    File2 = dir_path+'label_val_4_noLPS.txt'
    xy_label2 = np.loadtxt(File2,delimiter='\t', dtype=np.int64, encoding='utf-16')
    xy_label = np.concatenate((xy_label1,xy_label2),axis=0)

    device = "cpu"
    for i in range(4):
        print(i)
        index = np.where(xy_label==i)[0]
        xy_i = xy[index]
        xy_label_i = xy_label[index]
        map_i = np.zeros((221, 222))
        k = 0
        for j in range(index.shape[0]):
            print(j)
            sample = xy_i[j]
            label = xy_label_i[j]
            a = 221# 221
            b = 222
            sample = np.reshape(sample, (1, 1, a, b))
            sample=torch.from_numpy(sample).float()
            yb_pre = net(sample)
            yb_pre = yb_pre.detach()
            yb_pre = torch.argmax(yb_pre, dim = 1)
            yb_pre = yb_pre.numpy()[0]
            if yb_pre == label:
                vis = SaliencyMap(net, device)
                outs = vis(sample, layer = None, guide=True)
                outs = outs.numpy()
                map_i = map_i + outs
                k += 1
        map_i = map_i/k
        
        plt.imshow(map_i)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join("map_{}.png".format(i)))
        plt.show()    
        
    
    # Visualize the raw CAM
    plt.imshow(outs)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
