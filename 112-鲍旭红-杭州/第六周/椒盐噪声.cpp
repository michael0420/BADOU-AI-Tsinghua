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





//为图像添加高斯噪声 rate:椒盐噪声的比例
void addSaltPeperNoise(Mat srcImg, Mat& dstImg, double rate)
{
	
	dstImg = srcImg.clone();
	int channels = srcImg.channels();//获取图像的通道
	int rows = srcImg.rows;			//图像的行数
	int cols = srcImg.cols;//图像的总列数
	int salt_peper_num = rows * cols * rate;
	int val = 0;
	int n = 0;
	srand((unsigned)time(NULL));
	int i = 0;
	int j = 0;
	for (int k = 0; k < salt_peper_num; k++)
	{
		n = rand() % 2;
		i = rand() % rows;
		j = rand() % cols;
		if (channels == 1)
		{
			if (n == 0)
			{
				dstImg.ptr<uchar>(i)[j] = 0;
			}
			if (n == 1)
			{
				dstImg.ptr<uchar>(i)[j] = 255;
			}
		}
		else
		{
			if (n == 0)
			{
				dstImg.ptr<Vec3b>(i)[j][0] = 0;
				dstImg.ptr<Vec3b>(i)[j][1] = 0;
				dstImg.ptr<Vec3b>(i)[j][2] = 0;
			}
			if (n == 1)
			{
				dstImg.ptr<Vec3b>(i)[j][0] = 255;
				dstImg.ptr<Vec3b>(i)[j][1] = 255;
				dstImg.ptr<Vec3b>(i)[j][2] = 255;
			}
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
	addSaltPeperNoise(img, dstImg,0.01);
	imshow("dstImg0", dstImg);
	Mat grayImg;
	cvtColor(img, grayImg, CV_BGR2GRAY);
	imshow("grayImg", grayImg);
	addSaltPeperNoise(grayImg, dstImg, 0.01);
	imshow("dstImg1", dstImg);
	waitKey(0);
}