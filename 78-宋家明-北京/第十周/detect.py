import torch
from torch import nn
from torch_alexnet import alexnet
from torchvision import datasets
from torchvision.transforms import Compose,ToTensor,Resize
from torch.utils.data import DataLoader
import cv2


def detect(test_loader,model,device):
    """
    预测
    """
    model.eval()
    with torch.no_grad():
        for batch,(x,y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            print(y)
            print(pred.argmax(0))




def main(device,class_num,model_path):
    
    test_data = datasets.CIFAR10(root='../data',train=False,transform=Compose([Resize((224,224)),ToTensor()]),download=True)
    test_loader = DataLoader(test_data,batch_size=8)

    model = alexnet(class_num).to(device)
    model.load_state_dict(torch.load(model_path))
    detect(test_loader,model,device)                

    print('sucess run!')




if __name__=='__main__':

    """
    深度学习分类预测code
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_num = 10
    model_path = './alexnetmodel.pth'
    main(device,class_num,model_path)
