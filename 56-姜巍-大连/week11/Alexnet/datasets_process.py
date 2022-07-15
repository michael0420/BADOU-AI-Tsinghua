"""
使用PyTorch加载自己的数据集
"""
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """
    需要写一个继承自torch.utils.data中Dataset类，并修改其中的__init__方法、__getitem__方法、__len__方法。默认加载的都是图片。
    """

    def __init__(self, txt_path, transform=None, target_transform=None):
        """
        得到一个包含数据和标签的list，每个元素能找到图片位置和其对应标签。
        :param txt_path: 数据集标签索引文件的路径
        :param transform: 对数据集做变换操作
        """
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))  # 类别转为整型int
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        """
        得到每个元素的图像像素矩阵和标签。
        :param index: 索引
        :return: img和label
        """
        file_name, label = self.imgs[index]
        img = Image.open(file_name).convert('RGB')
        # img = Image.open(file_name)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
