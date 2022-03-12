from torch import nn
from torchsummary import summary


def bottlenet(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=3,
            stride=stride, padding=1, groups=in_channels),
        nn.BatchNorm2d(in_channels, eps=0.001),
        nn.ReLU6(inplace=True),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        nn.BatchNorm2d(out_channels, eps=0.001),
        nn.ReLU6(inplace=True),
    )


class MobileNet_V1(nn.Module):
    def __init__(self, num_classes=1000, **kwargs):
        super(MobileNet_V1, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.block2_1 = bottlenet(in_channels=32, out_channels=64, stride=1)
        self.block2_2 = bottlenet(in_channels=64, out_channels=128, stride=2)
        self.block2_3 = bottlenet(in_channels=128, out_channels=128, stride=1)
        self.block2_4 = bottlenet(in_channels=128, out_channels=256, stride=2)
        self.block2_5 = bottlenet(in_channels=256, out_channels=256, stride=1)
        self.block2_6 = bottlenet(in_channels=256, out_channels=512, stride=2)

        self.block3_1 = bottlenet(in_channels=512, out_channels=512, stride=1)
        self.block3_2 = bottlenet(in_channels=512, out_channels=512, stride=1)
        self.block3_3 = bottlenet(in_channels=512, out_channels=512, stride=1)
        self.block3_4 = bottlenet(in_channels=512, out_channels=512, stride=1)
        self.block3_5 = bottlenet(in_channels=512, out_channels=512, stride=1)

        self.block4_1 = bottlenet(in_channels=512, out_channels=1024, stride=2)
        self.block4_2 = bottlenet(in_channels=1024, out_channels=1024, stride=1)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block2_3(x)
        x = self.block2_4(x)
        x = self.block2_5(x)
        x = self.block2_6(x)

        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        x = self.block3_5(x)
        print(x.shape)
        x = self.block4_1(x)
        print(x.shape)
        x = self.block4_2(x)
        print(x.shape)
        x = self.avg_pool(x)

        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)

        return output

model = MobileNet_V1(1000)
summary(model, input_size=(3, 224, 224), batch_size=1)
