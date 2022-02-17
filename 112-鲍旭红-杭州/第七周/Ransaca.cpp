#include<iostream>
#include<math.h>
#include"opencv2/opencv.hpp"
#include"opencv2/highgui/highgui.hpp"
#include"opencv2\imgproc\imgproc.hpp"


using namespace std;
using namespace cv;
#define WITHOUT_POINT_NUM 50
#define MAX_POINT_NUM 220

RNG rng((unsigned)time(NULL));


/*最小二乘法*/
void least_squests(vector<Point2f> a, float *k, float *b)
{
	int n = a.size();
	double sum_x = 0;
	double sum_y = 0;
	double sum_xy = 0;
	double sum_xx = 0;
	/*1、计算累加和*/
	for (int i = 0; i < n; i++)
	{
		sum_x += a[i].x;
		sum_y += a[i].y;
		sum_xy += a[i].x * a[i].y;
		sum_xx += a[i].x * a[i].x;
	}
	/*2、求斜率k和截距b*/
	*k = (sum_xy * n - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
	*b = (sum_y - *k * sum_x) / n;
}


void ransaca(vector<Point2f> a, float *k, float *b)
{
	const int N = 20;
	const int M = 300;
	const int THRESHOLD = 50;
	int within_group[M] = {0};
	float within_max = 0;
	for (int n = 0; n < M; n++)
	{
		/*1、从a中随机取N个点*/
		vector<Point2f>p_b;
		for (int i = 0; i < N; i++)
		{
			int j = rng.uniform(1, MAX_POINT_NUM);
			//cout << j << " ";
			p_b.push_back(a[j]);
		}
		/*2、用最小二乘法拟合第一步随机取的N个点*/
		float k_tmp = 0, b_tmp = 0;
		least_squests(p_b, &k_tmp, &b_tmp);
		
		/*3、计算生成的参数有多少个内群点*/
		for (int m = 0; m < MAX_POINT_NUM; m++)
		{
			if (abs(a[m].y - (k_tmp * a[m].x + b_tmp)) < THRESHOLD)
			{
				within_group[n]++;
			}
		}
		cout << "第" << n + 1 << "次拟合的参数：" << "k = " << k_tmp << " b = " << b_tmp << " 内群点数量:"<<within_group[n]<<endl;
		/*4、找出最大数量内群点的k和b*/
		if (within_max < within_group[n])
		{
			within_max = within_group[n];
			*k = k_tmp;
			*b = b_tmp;
		}
	}
}





int main(void)
{
	vector<Point2f> a;
	Point2f p_a;
	for (int i = 0; i < WITHOUT_POINT_NUM; i++)
	{
		p_a.x = rng.uniform(1, 800);
		p_a.y = rng.uniform(1, 100);
		a.push_back(p_a);
	}
	for (int i = 0; i < MAX_POINT_NUM - WITHOUT_POINT_NUM; i++)
	{
		p_a.x = rng.uniform(1,800);
		p_a.y = rng.uniform(500, 600);
		a.push_back(p_a);
	}
	float k = 0, b = 0;
	least_squests(a, &k, &b);
	cout << "k = " << k << " b = " << b << endl;
	Mat img = Mat::zeros(Size(800, 800), CV_8UC3);
	for (int i = 0; i < a.size(); i++)
	{
		circle(img, a[i], 1, Scalar(201, 34, 234), CV_FILLED);
	}
	Point2f a1,a2;
	a1.x = 34.8;
	a2.x = 787.9;
	a1.y = k * a1.x + b;
	a2.y = k * a2.x + b;
	line(img, a1, a2, Scalar(0, 0, 255));
	putText(img, "least squests", a1, 3, 1.5, Scalar(0, 0, 255));
	ransaca(a, &k, &b);
	a1.x = 34.8;
	a2.x = 787.9;
	a1.y = k * a1.x + b;
	a2.y = k * a2.x + b;
	line(img, a1, a2, Scalar(255, 255, 255));
	
	putText(img, "ransaca", a1, 3, 1.5, Scalar(255, 255, 255));
	imshow("img",img);
	waitKey(0);
	system("pause");
}