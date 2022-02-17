#include "opencv2/opencv.hpp"



using namespace std;
using namespace cv;



void img2Gray(Mat img,Mat& dst)
{
	vector<Mat>bgr;
	Mat r, g, b;
	split(img, bgr);
	b = bgr[0];
	g = bgr[1];
	r = bgr[2];
	dst = b*0.11 + g*0.59 + r*0.3;
}


#if 1
int main(void)
{
	string path = "D:\\八斗清华班\\【2】数学基础&数字图像\\八斗学院作业\\lena_c.bmp";
	Mat srcImg = imread(path);
	Mat grayImg;
	img2Gray(srcImg,grayImg);
	imshow("srcImg", srcImg);
	imshow("grayImg", grayImg);
	waitKey(0);
}
#endif
