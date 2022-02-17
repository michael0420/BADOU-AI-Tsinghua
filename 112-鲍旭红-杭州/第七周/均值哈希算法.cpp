#include"opencv2/opencv.hpp"
#include"opencv2/highgui/highgui.hpp"
#include"opencv2\imgproc\imgproc.hpp"



using namespace cv;
using namespace std;



string avarge_hash(Mat src)
{
	Mat resize_8_8;
	resize(src, resize_8_8, Size(8, 8));
	Mat grayImg;
	if (src.channels() != 1)
	{
		cvtColor(resize_8_8, grayImg, CV_BGR2GRAY);
	}
	int channels = grayImg.channels();
	int rows = grayImg.rows;
	int cols = grayImg.cols * channels;//图像总列数
	/*判断图像的连续性*/
	if (grayImg.isContinuous())
	{
		/*如果图像连续，相当于只遍历一个一维数组*/
		cols *= rows;
		rows = 1;
	}
	/*求灰度图所有像素之和*/
	uint32_t sum = 0;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			uchar val = grayImg.ptr<uchar>(i)[j];
			sum += val;
		}
	}
	/*求均值*/
	uchar avgVal = sum / (rows * cols);
	string hash = "";
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			uchar val = grayImg.ptr<uchar>(i)[j];
			if (val > avgVal)
			{
				hash += "1";
			}
			else
			{
				hash += "0";
			}
		}
	}
	return hash;
}



int main(void)
{
	string path1 = "F:\\八斗清华班\\lena\\temp\\lena.bmp";
	string path2 = "F:\\八斗清华班\\lena\\temp\\lena.jpg";
	Mat srcImg1 = imread(path1);
	Mat srcImg2 = imread(path2);
	string hash1 = avarge_hash(srcImg1);
	string hash2 = avarge_hash(srcImg2);
	cout << hash1 <<"\n"<< hash2 << endl;
	int similarity_rate = 0;
	for (int i = 0; i < hash1.length(); i++)
	{
		if (hash1[i] == hash2[i])
		{
			similarity_rate++;
		}
	}
	cout << "两张图片的相似率: " << similarity_rate <<"%"<< endl;
	system("pause");
}