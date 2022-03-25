import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


class mnist_net(nn.Module):
    def __init__(self):
        super(mnist_net,self).__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
                nn.Linear(28*28,512),
                nn.ReLU(),
                nn.Linear(512,256),
                nn.ReLU(),
                nn.Linear(256,10),
                )
    def forward(self,x):
        x = self.flatten(x)
        x = self.net(x)
       
        return x

def train(train_loader,device,model,loss_fun,optim,epochs):

    model.train()
    for epoch in range(epochs):
        for batch,(x,y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)

            loss = loss_fun(pred,y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            print(f'epoch:{epoch} batch:{batch} loss:{loss:>5f}')
    print('train sucess')

def test(test_loader,device,model,loss_fun):

    model.eval()
    batch_num = len(test_loader)
    loss = 0
    with torch.no_grad():
        for batch,(x,y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)

            loss = loss_fun(pred,y)
    loss = loss/batch_num
    print(f'test:   loss:{loss:>5f}')






def main(batch_size,device,epochs):

    train_data = datasets.MNIST(root='../data',train=True,download=True,transform=ToTensor())
    test_data = datasets.MNIST(root='../data',train=False,download=True,transform=ToTensor())

    train_loader = DataLoader(train_data,batch_size=batch_size)
    test_loader = DataLoader(test_data,batch_size=batch_size)

    model = mnist_net().to(device)

    loss_fun = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(),lr=1e-3)

    train(train_loader,device,model,loss_fun,optim,epochs)
    test(test_loader,device,model,loss_fun)
    
    torch.save(model.state_dict(),'mnist_net.pth')
    print('save sucess')
    

if __name__=='__main__':

    """
    torch 推理mnist数据集
    
    """
    batch_size = 64
    epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(batch_size,device,epochs)
