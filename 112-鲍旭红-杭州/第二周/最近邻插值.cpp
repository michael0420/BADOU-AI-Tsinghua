#include "opencv2/opencv.hpp"



using namespace std;
using namespace cv;


//最近邻插值
//sx、sy为缩放因子
void nearest_neighbor(Mat img, Mat& dst, double sx, double sy)
{
	uchar *pImg = img.data;
	int channel = img.channels();
	int dst_cols = round(img.cols * sx);
	int dst_rows = round(img.rows * sy);
	dst = Mat(Size(dst_cols, dst_rows), img.type());
	uchar *pDst = dst.data;
	int step = img.step / sizeof(uchar);
	uchar *src_rows_ptr_1 = NULL;
	uchar *dst_rows_ptr_1 = NULL;
	Vec3b *src_rows_ptr_3 = NULL;
	Vec3b *dst_rows_ptr_3 = NULL;
	for (int i = 0; i < dst_rows; i++)
	{
		int src_rows = round(i / sy);
		/*防止越界*/
		if (src_rows < 0)
		{
			src_rows = 0;
		}
		if (src_rows >(img.rows - 1))
		{
			src_rows = img.rows - 1;
		}
		if (img.channels() == 1)
		{
			src_rows_ptr_1 = img.ptr<uchar>(src_rows);
			dst_rows_ptr_1 = dst.ptr<uchar>(i);
		}
		else
		{
			src_rows_ptr_3 = img.ptr<Vec3b>(src_rows);
			dst_rows_ptr_3 = dst.ptr<Vec3b>(i);
		}
		for (int j = 0; j < dst_cols; j++)
		{
			int src_cols = round(j / sx);
			/*防止越界*/
			if (src_cols < 0)
			{
				src_cols = 0;
			}
			if (src_cols >(img.cols - 1))
			{
				src_cols = img.cols - 1;
			}
			int src_channels = src_cols * channel;
			int dst_channels = j * channel;
			if (img.channels() == 1)
			{
				dst_rows_ptr_1[j] = src_rows_ptr_1[src_cols];
			}
			else
			{
				dst_rows_ptr_3[j][0] = src_rows_ptr_3[src_cols][0];
				dst_rows_ptr_3[j][1] = src_rows_ptr_3[src_cols][1];
				dst_rows_ptr_3[j][2] = src_rows_ptr_3[src_cols][2];
			}
		}
	}
}


#if 0
int main(void)
{
	string path = "D:\\八斗清华班\\【2】数学基础&数字图像\\八斗学院作业\\lena_c.bmp";
	Mat srcImg = imread(path);
	Mat grayImg;
	Mat dstImg;
	cvtColor(srcImg, grayImg, CV_BGR2GRAY);
	nearest_neighbor(grayImg, dstImg, 0.5, 0.5);
	imshow("srcImg", srcImg);
	imshow("grayImg", dstImg);
	waitKey(0);
}
#endif
