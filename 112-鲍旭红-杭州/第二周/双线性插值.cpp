#include "opencv2/opencv.hpp"



using namespace std;
using namespace cv;


//双线性插值
//sx、sy为缩放因子
void Inter_Linear(Mat img, Mat& dst, double sx, double sy)
{
	uchar *pImg = img.data;
	int channel = img.channels();
	int dst_cols = round(img.cols * sx);
	int dst_rows = round(img.rows * sy);
	dst = Mat(Size(dst_cols, dst_rows), img.type());
	uchar *pDst = dst.data;
	int step = img.step / sizeof(uchar);
	uchar *src_rows_ptr_1_l = NULL;
	uchar *src_rows_ptr_1_r = NULL;
	uchar *dst_rows_ptr_1 = NULL;
	Vec3b *src_rows_ptr_3_l = NULL;
	Vec3b *src_rows_ptr_3_r = NULL;
	Vec3b *dst_rows_ptr_3 = NULL;
	for (int i = 0; i < dst_rows; i++)
	{
		/*几何中心对齐*/
		double index_i = (i + 0.5) / sx - 0.5;
		/*防止越界*/
		if (index_i < 0)
		{
			index_i = 0;
		}
		if (index_i > (img.rows - 1))
		{
			index_i = img.rows - 1;
		}
		/*相邻4像素的坐标*/
		int i1 = floor(index_i);
		int i2 = ceil(index_i);
		//u为传到浮点型坐标的小数部分
		double u = index_i - i1;
		if (img.channels() == 1)
		{
			src_rows_ptr_1_l = img.ptr<uchar>(i1);
			src_rows_ptr_1_r = img.ptr<uchar>(i2);
			dst_rows_ptr_1 = dst.ptr<uchar>(i);
		}
		else
		{
			src_rows_ptr_3_l = img.ptr<Vec3b>(i1);
			src_rows_ptr_3_r = img.ptr<Vec3b>(i2);
			dst_rows_ptr_3 = dst.ptr<Vec3b>(i);
		}
		for (int j = 0; j < dst_cols; j++)
		{
			/*几何中心对齐*/
			double index_j = (j + 0.5) / sy - 0.5;
			/*防止越界*/
			if (index_j < 0)
			{
				index_j = 0;
			}
			if (index_j > (img.cols - 1))
			{
				index_j = img.cols - 1;
			}
			/*相邻4像素的坐标*/
			int j1 = floor(index_j);
			int j2 = ceil(index_j);
			//v为传到浮点型坐标的小数部分
			double v = index_j - j1;
			if (img.channels() == 1)
			{
				dst_rows_ptr_1[j] = (1 - u) * (1-v) * src_rows_ptr_1_l[j1] + (1 - u) * v * src_rows_ptr_1_l[j2] + u * (1 - v) * src_rows_ptr_1_r[j1] + u * v * src_rows_ptr_1_r[j2];
				//dst_rows_ptr_1[j] = (1 - u) * (1 - v) * img.at<uchar>(i1,j1) + (1 - u) * v * img.at<uchar>(i1, j2) + u * (1 - v) * img.at<uchar>(i2, j1) + u * v * img.at<uchar>(i2, j2);
			}
			else
			{
				dst_rows_ptr_3[j][0] = (1 - u) * (1 - v) * src_rows_ptr_3_l[j1][0] + (1 - u) * v * src_rows_ptr_3_l[j2][0] + u * (1 - v) * src_rows_ptr_3_r[j1][0] + u * v * src_rows_ptr_3_r[j2][0];
				dst_rows_ptr_3[j][1] = (1 - u) * (1 - v) * src_rows_ptr_3_l[j1][1] + (1 - u) * v * src_rows_ptr_3_l[j2][1] + u * (1 - v) * src_rows_ptr_3_r[j1][1] + u * v * src_rows_ptr_3_r[j2][1];
				dst_rows_ptr_3[j][2] = (1 - u) * (1 - v) * src_rows_ptr_3_l[j1][2] + (1 - u) * v * src_rows_ptr_3_l[j2][2] + u * (1 - v) * src_rows_ptr_3_r[j1][2] + u * v * src_rows_ptr_3_r[j2][2];
			}
		}
	}
}


#if 1
int main(void)
{
	string path = "D:\\八斗清华班\\【2】数学基础&数字图像\\八斗学院作业\\lena_c.bmp";
	Mat srcImg = imread(path);
	Mat grayImg;
	Mat dstImg;
	cvtColor(srcImg, grayImg, CV_BGR2GRAY);
	Inter_Linear(srcImg, dstImg, 0.8, 0.8);
	imshow("srcImg", srcImg);
	imshow("grayImg", dstImg);
	waitKey(0);
}
#endif
