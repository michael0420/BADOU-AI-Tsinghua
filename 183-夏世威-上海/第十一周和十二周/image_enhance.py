import shutil
import PIL.Image as Image
import os
from torchvision import transforms as transforms

# 运行image_enhance.py,通过图像增强,这里简单使用水平竖直翻转,将400张图片扩大到1200张
# PALM-Training400 -> PALM-Training1200
shutil.copytree("PALM-Training400", "PALM-Training1200")
for dir, dir_list, file_list in os.walk("PALM-Training1200"):
    for file in file_list:
        file_path = os.path.join(dir, file)
        im = Image.open(file_path)
        new_im = transforms.RandomHorizontalFlip(p=1)(im)  # p表示概率
        files = str.split(file_path, ".")
        new_im.save(files[0] + "-1.jpg")
        new_im = transforms.RandomVerticalFlip(p=1)(im)
        new_im.save(files[0] + "-2.jpg")
