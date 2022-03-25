from nets_src import alexnet
from nets_src import vgg
from nets_src import googlenet
from nets_src import resnet
from nets_src import mobilenet

nets = {
    "AlexNet": {"model": alexnet.AlexNet(), "path": "./nets_data/alexnet"},
    "VGG": {"model": vgg.VGG(), "path": "./nets_data/vgg"},
    "GoogLeNet": {"model": googlenet.GoogLeNet(), "path": "./nets_data/googlenet"},
    "ResNet": {"model": resnet.ResNet(), "path": "./nets_data/resnet"},
    "MobileNet": {"model": mobilenet.MobileNet(), "path": "./nets_data/mobilenet"},
}
