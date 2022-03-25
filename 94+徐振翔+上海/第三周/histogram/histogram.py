# 多种方法实现作业：1.使用自制函数 2.使用库1 3.使用库2 ......
from histogram_model import histogram
import numpy as np
import cv2
from matplotlib import pyplot as plt

#####################################################################
#           自制函数
#####################################################################
# 获取灰度图像
img = cv2.imread("lenna.png", cv2.IMREAD_COLOR)
# IMREAD_COLOR = 1,默认值也为1，默认输入3通道图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 灰度图像直方图均衡化
dst = histogram(gray)
# 计算像素级直方图
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
'''
# calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) -> hist
输入参数解析
images:输入的图像的指针，可以是多幅图像，所有的图像必须有同样的深度（CV_8U or CV_32F）。同时一副图像可以有多个channes。
channels：用来计算直方图的channes的数组。比如输入是2副图像，第一副图像有0，1，2共三个channel，第二幅图像只有0一个channel，
          那么输入就一共有4个channes，如果int channels[3] = {3, 2, 0}，那么就表示是使用第二副图像的第一个通道和第一副图像的第2和第0个通道来计算直方图。
mask：掩码,如果mask不为空，那么它必须是一个8位（CV_8U）的数组，并且它的大小的和arrays[i]的大小相同，值为1的点将用来计算直方图。
histSize:在每一维上直方图的个数。简单把直方图看作一个一个的竖条的话，就是每一维上竖条的个数。
ranges:用来进行统计的范围。
accumulate:Accumulation flag. If it is set, the histogram is not cleared in the beginning
            when it is allocated. This feature enables you to compute a single histogram from several
            sets of arrays, or to update the histogram in time.  
            是否累加。如果为true，在下次计算的时候不会首先清空hist。这个地方我是这样理解的，不知道有没有错，
'''

# plt.figure()
# plt.hist(dst.ravel(), 256)
# plt.show()

# 两个直方图一起显示
# 两个直方图，无重叠条
plt.hist([img.ravel(), dst.ravel()], 256, label=['src', 'dst'])
# 创建图例
plt.legend(loc='upper left')
plt.show()
# 两个直方图，有重叠条
plt.hist(img.ravel(), 256, alpha=0.5, label='a')
plt.hist(dst.ravel(), 256, alpha=0.5, label='b')
plt.legend(loc="upper left")
plt.show()
'''
hist(x, bins=None, range=None, density=False, weights=None,
        cumulative=False, bottom=None, histtype='bar', align='mid',
        orientation='vertical', rwidth=None, log=False, color=None,
        label=None, stacked=False, *, data=None, **kwargs):
常用参数解释：
x: 作直方图所要用的数据，必须是一维数组；多维数组可以先进行扁平化再作图；必选参数；
bins: 直方图的柱数，即要分的组数，默认为10；
range：元组(tuple)或None；剔除较大和较小的离群值，给出全局范围；如果为None，则默认为(x.min(), x.max())；即x轴的范围；
density：布尔值。如果为true，则返回的元组的第一个参数n将为频率而非默认的频数；
weights：与x形状相同的权重数组；将x中的每个元素乘以对应权重值再计数；如果normed或density取值为True，则会对权重进行归一化处理。这个参数可用于绘制已合并的数据的直方图；
cumulative：布尔值；如果为True，则计算累计频数；如果normed或density取值为True，则计算累计频率；
bottom：数组，标量值或None；每个柱子底部相对于y=0的位置。如果是标量值，则每个柱子相对于y=0向上/向下的偏移量相同。如果是数组，则根据数组元素取值移动对应的柱子；即直方图上下便宜距离；
histtype：{‘bar’, ‘barstacked’, ‘step’, ‘stepfilled’}；'bar’是传统的条形直方图；'barstacked’是堆叠的条形直方图；'step’是未填充的条形直方图，只有外边框；‘stepfilled’是有填充的直方图；当histtype取值为’step’或’stepfilled’，rwidth设置失效，即不能指定柱子之间的间隔，默认连接在一起；
align：{‘left’, ‘mid’, ‘right’}；‘left’：柱子的中心位于bins的左边缘；‘mid’：柱子位于bins左右边缘之间；‘right’：柱子的中心位于bins的右边缘；
orientation：{‘horizontal’, ‘vertical’}：如果取值为horizontal，则条形图将以y轴为基线，水平排列；简单理解为类似bar()转换成barh()，旋转90°；
rwidth：标量值或None。柱子的宽度占bins宽的比例；
log：布尔值。如果取值为True，则坐标轴的刻度为对数刻度；如果log为True且x是一维数组，则计数为0的取值将被剔除，仅返回非空的(frequency, bins, patches）；
color：具体颜色，数组（元素为颜色）或None。
label：字符串（序列）或None；有多个数据集时，用label参数做标注区分；
stacked：布尔值。如果取值为True，则输出的图为多个数据集堆叠累计的结果；如果取值为False且histtype=‘bar’或’step’，则多个数据集的柱子并排排列；
normed: 是否将得到的直方图向量归一化，即显示占比，默认为0，不归一化；不推荐使用，建议改用density参数；
edgecolor: 直方图边框颜色；
alpha: 透明度；

返回值（用参数接收返回值，便于设置数据标签）：
n：直方图向量，即每个分组下的统计值，是否归一化由参数normed设定。当normed取默认值时，n即为直方图各组内元素的数量（各组频数）；
bins: 返回各个bin的区间范围；
patches：返回每个bin里面包含的数据，是一个list。
其他参数与plt.bar()类似。
'''

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)

#####################################################################
#          使用库1
#####################################################################












