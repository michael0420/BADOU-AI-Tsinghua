# 复现经典的图像识别网络

模型简单理解

* AlexNet 卷积神经网络
* VGG 网络加深
* GoogLeNet 网络加宽,利用Inception结构
* ResNet 引入残差结构
* MobileNet 深度可分离卷积

注意:因为相对于网络来说,数据集很小,所以在训练过程中非常容易产生过拟合,解决办法可以考虑增加数据集或者减小模型
在源码中,为了准确度,缩小了模型,注释掉了部分隐藏层

# 数据集介绍

眼疾数据集  
训练集PALM-Training400:400张图,有三种图片

* 病理性近视（PM）：文件名以P开头
* 非病理性近视（non-PM）：
    * 高度近视（high myopia）：文件名以H开头
    * 正常眼睛（normal）：文件名以N开头

  但我们这里分两类,病理性为正样本,标签为1;非病理性的为负样本标签为0

训练集PALM-Training1200:1200张图,运行image_enhance.py,通过图像增强,这里简单使用水平竖直翻转,将400张图片扩大到1200张  
通过运行image_enhance.py,获取

验证集PALM-Validation400:400张图,标签在labels.csv里面

# 代码目录层级

* src 代码及模型文件都在里面
    * data_loader.py 从数据集中分批加载数据
    * train.py 训练模型,可直接运行
    * predict.py 测试模型,可直接运行
    * nets_data 训练好的模型保存位置,由于模型文件较大,模型文件都被删了
    * nets_src 模型的设计代码,包括AlexNet VGG GoogleNet ResNet
    * nets_config 主要显示有哪些模型及保存位置

# 使用
1. 先运行image_enhance.py,获取PALM-Training1200
2. 运行train.py 由于模型文件较大,都被删了,所以必须要先运行train.py,不过普通的CPU也能几分钟完成,不要担心
3. 运行predict.py,运行测试集,查看模型效果
