import torch
from torch import nn
from torch_alexnet import alexnet
from torchvision import datasets
from torchvision.transforms import Compose,ToTensor,Resize
from torch.utils.data import DataLoader
import cv2


def train(train_loader,model,loss_fun,optim,epochs,device):
    """
    训练
    """
    model.train()
    for epoch in range(epochs):
        for batch,(x,y) in enumerate(train_loader):
            
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fun(pred,y)
            optim.zero_grad()
            loss.backward()
            optim.step()


            print(f'epoch:{epoch} batch:{batch} loss:{loss}')
    print('train success')

    torch.save(model.state_dict(),'alexnetmodel.pth')
    print('save success')

def main(device,epochs,lr,class_num):
    
    train_data = datasets.CIFAR10(root='../data',train=True,transform=Compose([Resize((224,224)),ToTensor()]),download=True)
    train_loader = DataLoader(train_data,batch_size=8)

    model = alexnet(class_num).to(device)
    loss_fun = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(),lr=lr)

    train(train_loader,model,loss_fun,optim,epochs,device)                

    print('sucess run!')




if __name__=='__main__':

    """
    深度学习训练code
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 20
    lr = 1e-3
    class_num = 10
    main(device,epochs,lr,class_num)
