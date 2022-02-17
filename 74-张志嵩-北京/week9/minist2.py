# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 22:47:25 2022

@author: Administrator
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import Alexnet, resnet18, resnet50, vgg11, vgg11_bn, vgg16, inception_v3

from tqdm import tqdm

def main():
    epochs = 100
    model_name = 'inception'#'vgg'#"alexnet"#'resnet'
    if model_name == 'inception':
        transform = transforms.Compose([transforms.Resize((299,299)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5],[1])
            ])
    else:
        transform = transforms.Compose([transforms.Resize((227, 227)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5], [1])
                                        ])
    data_train = datasets.MNIST(root="./dataset", transform=transform, train =True,download=True)
    data_test = datasets.MNIST(root='./dataset', transform=transform, train=False)
    
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size=64,
                                                    shuffle=True
                                                    )
    data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                   batch_size=64,
                                                   )
    if model_name == 'alexnet':
        model = Alexnet(num_classes=10)
    elif model_name == 'resnet':
        model = resnet18(num_classes=10)
        #model = resnet50(num_classes=10)
    elif model_name == 'vgg':
        model = vgg11(num_classes=10)
    elif model_name == 'inception':
        model = inception_v3(num_classes=10)
    print(model)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    if torch.cuda.is_available():
        model.cuda()
    cost = nn.CrossEntropyLoss()
    cost2 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-1)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(data_loader_train), epochs=epochs)
    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0
        print("Epoch{}/{}".format(epoch,epochs))
        print("-"*10)
        pbar = tqdm(data_loader_train)
        for data in pbar:
            x,y=data
            x = torch.cat((x,x,x),1)
            if torch.cuda.is_available():
                x,y = x.cuda(), y.cuda()
            outputs = model(x)
            if model_name == 'inception':
                outputs = outputs[0]
            _, pred = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss = cost(outputs,y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.data.cpu().numpy()
            running_correct += torch.sum(pred==y)
            pbar.set_description("Processing loss is %.4f, correct is %d"%(loss.data.cpu().numpy(),torch.sum(pred==y)))
        testing_correct = 0
        for i,data in enumerate(data_loader_test):
            x,y = data
            x = torch.cat((x,x,x),1)
            if torch.cuda.is_available():
                x,y = x.cuda(), y.cuda()
            outputs = model(x)
            _, pred = torch.max(outputs,1)
            testing_correct += torch.sum(pred==y)
        print("\n")
        print("Loss is :{:.5f}, Train Accuracy is {:.4f}%, Test Accuracy is: {:.4f}%, lr is: {:.4f}".format(running_loss/len(data_train),100*running_correct.data.cpu().numpy()/len(data_train),100*testing_correct.data.cpu().numpy()/len(data_test),optimizer.state_dict()['param_groups'][0]['lr']))

if __name__ == "__main__":
    main()