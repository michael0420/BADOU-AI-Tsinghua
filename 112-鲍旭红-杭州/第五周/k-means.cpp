#include <fstream>
#include <iostream>
#include <stdlib.h>
#include<vector>
#include<math.h>
#include <algorithm> // std::move_backward
#include <random> // std::default_random_engine
#include <chrono> // std::chrono::system_clock
#include "opencv2/opencv.hpp"


using namespace std;
using namespace cv;

//#define MAX_NUM 10


//求两点间距离
float two_point_distance(Point2f a, Point2f b)
{
	float lenth = 0;
	lenth = sqrtf((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
	return lenth;
}

//找最小值序号,质心除外
int get_min_value(vector<float> data)
{
	float min_value = 999999.99;
	int min_index = 0;
	for (int i = 0; i < data.size(); i++)
	{
		if (min_value > data[i])
		{	
			min_value = data[i];
			min_index = i;
		}
	}
	return min_index;
}




//求质心
vector<Point2f> get_center_point(vector<vector<Point2f>> src,  int K)
{
	vector<Point2f> center;
	Point2f avg;
	Point2f sum = {0,0};
	for (int i = 0; i < K; i++)
	{
		for (int j = 0; j < src[i].size(); j++)
		{
			sum.x += src[i][j].x;
			sum.y += src[i][j].y;
			//cout << "x = "<<src[i][j].x <<" y = "<< src[i][j].y << endl;
		}
		avg.x = sum.x / src[i].size();
		avg.y = sum.y / src[i].size();
		/*cout << "avg = " << avg << endl;
		cout << "sum = " << sum << endl;*/
		center.push_back(avg);
	}
	return center;
}



void k_means(vector<Point2f> input, int K, vector<vector<Point2f>>* output)
{
	int index = 0;
	int min_index = 0;
	vector<int>center_index;
	vector<float>distance;
	vector<Point2f> center_point;
	center_point.resize(K);
	vector<vector<Point2f>> reslut,reslut_tmp;
	center_point.resize(K);
	reslut.resize(K);
	reslut_tmp.resize(K);
	int count = input.size() / K;
	/*打乱原始数据
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	shuffle(input.begin(), input.end(), default_random_engine(seed));
	cout << "\nshuffle后数据: " << endl;
	for (int i = 0; i < input.size(); i++)
	{
		cout << input[i] << " ,";
	}*/
	/*1、把点数据分成K个簇*/
	for (int i = 0; i < K; i++)
	{
		for (int j = 0; j < input.size(); j++)
		{
			if ((j >= count * i) && (j < count * (i + 1)))
			{
				reslut[i].push_back(input[j]);
			}
		}
	}
	while (true)
	{
		/*2、求各簇的质心*/
		reslut_tmp = reslut;
		center_point = get_center_point(reslut, K);
		//cout << "center_point: "<<center_point << endl;
		reslut.clear();
		reslut.resize(K);
		/*3、计算所有点到质心的距离，根据各点到质心的距离，重新分组*/
		for (int i = 0; i < input.size(); i++)
		{
			distance.clear();
			for (int j = 0; j < K; j++)
			{
				distance.push_back(two_point_distance(center_point[j], input[i]));
			}
			min_index = get_min_value(distance);
			reslut[min_index].push_back(input[i]);
		}
		/*4、当前后分组数据相等时，即表示计算完毕，退出循环*/
		if (reslut_tmp == reslut)
		{
			break;
		}
	}
	*output = reslut;
}



int main(void)
{
	vector<Point2f> point = { { 0.02f,0.06f },{ 1.07f,2.07f },{ 3.03f,1.01f },{ 8.04f,8.05f },{ 9.09f,10.07f },
	{ 10.06f,7.12f },{ 18.34f,78.12f },{ 14.03f,73.09f },{ 68.8f,88.2f },{ 104.063f,713.09f },{ 168.8f,868.2f } };
	cout << "待分组数据: " << endl;
	for (int i = 0; i < point.size(); i++)
	{
		cout << point[i] <<" ,";
	}
	
	int k = 3;
	vector<vector<Point2f>> reslut;
	k_means(point, k,&reslut);
	cout << "\n\nk = "<<k<<", k-means结果:" << endl;
	for (int i = 0; i < k; i++)
	{
		cout << "第"<<i<<"组结果:" << endl;
		for (int j = 0; j < reslut[i].size(); j++)
		{
			cout << reslut[i][j]<<" " << endl;
		}
	}
	system("pause");
}