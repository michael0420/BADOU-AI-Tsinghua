#include "opencv2/opencv.hpp"



using namespace std;
using namespace cv;



void myThreshold(Mat img, Mat& dst,int minV,int maxV,Scalar a,Scalar b)
{
	uchar *pSrc = img.data;
	int step = img.step / sizeof(uchar);
	/*建立一个和原图像相同大小的颜色设置为a背景图像*/
	dst = Mat(img.size(),img.type(),a);
	int channels = img.channels();//图片通道
	Vec3b *dst_rows_ptr_3 = NULL;
	for (int i = 0; i < img.rows; i++)
	{
		uchar *pData = dst.ptr(i);
		int i_step = i * step;
		for (int j = 0; j < img.cols; j++)
		{
			int j_channels = j*channels;
			int value = pSrc[i_step + j_channels];
			dst_rows_ptr_3 = dst.ptr<Vec3b>(i);
			/*满足阈值条件则像素点填充255*/
			if ((value > minV) && (value < maxV))
			{
				dst_rows_ptr_3[j][0] = b[0];
				dst_rows_ptr_3[j][1] = b[1];
				dst_rows_ptr_3[j][2] = b[2];
			}
		}
	}
}


#if 1
int main(void)
{
	string path = "D:\\八斗清华班\\【2】数学基础&数字图像\\八斗学院作业\\lena.bmp";
	Mat srcImg = imread(path);
	Mat grayImg;
	myThreshold(srcImg, grayImg,23,100,Scalar(230,223,23),Scalar(45,125,65));
	imshow("srcImg", srcImg);
	imshow("grayImg", grayImg);
	waitKey(0);
}
#endif
