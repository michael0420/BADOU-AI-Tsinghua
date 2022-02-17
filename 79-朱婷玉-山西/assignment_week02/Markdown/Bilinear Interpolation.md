# 双线性插值  Bilinear Interpolation

## 插值

图像领域插值常用在修改图像尺寸的过程，由旧的图像矩阵中的点计算新图像矩阵中的点并插入。

## 单线性插值

已知图中$P_1(x_1,y_1)$, $P_2(x_2,y_2)$两点，要计算$[x_1,x_2]$区间内某一位置$x$在直线上的$y$值。

![img](https://pic3.zhimg.com/80/v2-126fe08d8c3fa51caaea6416f692d516_1440w.jpg)

根据直线的两点式 $\frac{y-y_1}{x-x_1}=\frac{y_2-y_1}{x_2-x_1}$ 

即 $y=\frac{x_2-x}{x_2-x_1}y1+\frac{x-x_1}{x_2-x_1}y_2 (y_1,y_2为加权系数)$ 

## 双线性插值

![image-20211125232916536](F:\MH\Python\pythonProject\assignment\Markdown\image-20211125232916536.png)

### 在x方向上插值

$f(R_1)=f(x,y_1)\approx \frac{x_2-x}{x_2-x_1}f(Q_{11})+\frac{x-x_1}{x_2-x_1}f(Q_{21})$ 

$f(R_2)=f(x,y_2)\approx \frac{x_2-x}{x_2-x_1}f(Q_{12})+\frac{x-x_1}{x_2-x_1}f(Q_{22})$ 

分别以$Q_{11},Q_{21};Q_{12},Q_{22}$为已知两点做单线性插值。

### 在y方向上插值

$f(x,y)\approx \frac{y_2-y}{y_2-y_1}f(R_1)+\frac{y-y_1}{y_2-y_1}f(R_2)$ 

以$R_1,R_2$为已知两点，再次单线性插值。

### 合并

$\begin{split} f(x,y)&\approx \frac{y_2-y}{y_2-y_1}f(R_1)+\frac{y-y_1}{y_2-y_1}f(R_2)\\&=\frac{y_2-y}{y_2-y_1}(\frac{x_2-x}{x_2-x_1}f(Q_{11})+\frac{x-x_1}{x_2-x_1}f(Q_{21}))+\frac{y-y_1}{y_2-y_1}(\frac{x-x_1}{x_2-x_1}f(Q_{22})) \end{split}$ 



> 由于双线性插值所使用的4点$Q_{11},Q_{12},Q_{21},Q_{22}$均相邻，以上所有分母均为1。



## 存在的问题

### 坐标系的选择

#### 按比例对应

$srcX,srcY$为原图点坐标，$dstX,dstY$为变换后目标图像点坐标

$srcX=(dstX)*(srcWidth/dstWidth)$

$srcY=(dstY)*(srcHeight/dstHeight)$ 

#### 几何中心重合

让原图和目标图像几何中心重合，且目标图像每个像素等间隔，和两边均有一定边距。

