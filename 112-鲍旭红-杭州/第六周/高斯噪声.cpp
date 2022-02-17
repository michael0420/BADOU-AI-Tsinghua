#include <fstream>
#include <iostream>
#include <stdlib.h>
#include<vector>
#include <limits>
#include<math.h>
#include <algorithm> // std::move_backward
#include <random> // std::default_random_engine
#include <chrono> // std::chrono::system_clock
#include "opencv2/opencv.hpp"


using namespace std;
using namespace cv;

double generateGaussianNose(double mu, double sigma)
{
	//定义一个特别小的值
	const double epsilon = numeric_limits<double>::min();//返回目标数据类型能表示的最逼近1的正数和1的差的绝对值
	static double z0, z1;
	static bool flag = false;
	flag = !flag;
	//flag为假，构造高斯随机变量
	if (!flag)
	{
		return z1 * sigma + mu;
	}
	double u1, u2;
	//构造随机变量
	do 
	{
		u1 = rand()*(1.0 / RAND_MAX);
		u2 = rand()*(1.0 / RAND_MAX);
	} while (u1 < epsilon);
	//flag为真构造高斯随机变量X
	z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);
	z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI*u2);
	return z1 * sigma + mu;
}



//为图像添加高斯噪声
void addGaussianNoise(Mat srcImg, Mat& dstImg, double mu, double sigma)
{
	dstImg = srcImg.clone();
	int channels = srcImg.channels();//获取图像的通道
	int rows = srcImg.rows;			//图像的行数
	int cols = srcImg.cols * channels;//图像的总列数
	//判断图像的连续性
	if (srcImg.isContinuous())//判断矩阵是否连续，若连续，我们相当于只需要遍历一个一维数组 
	{
		cols *= rows;
		rows = 1;
	}
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			//添加高斯噪声
			double gaussian = generateGaussianNose(mu, sigma) * 32;
			//printf("gaussian = %.4f\n", gaussian);
			int val = srcImg.ptr<uchar>(i)[j] + gaussian;
			//printf("val = %d\n", val);
			if (val > 255)
			{
				val = 255;
			}
			if (val < 0)
			{
				val = 0;
			}
			dstImg.ptr<uchar>(i)[j] = (uchar)val;
		}
	}

}


int main(void)
{
	string img_path = "F:\\八斗清华班\\lena\\temp\\lena.bmp";
	Mat img = imread(img_path);
	if (!img.data)
	{
		return -1;
	}
	imshow("img", img);
	Mat dstImg;
	addGaussianNoise(img, dstImg,0.3,0.9);
	imshow("dstImg", dstImg);
	waitKey(0);
}