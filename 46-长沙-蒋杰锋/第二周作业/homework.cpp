#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
using namespace std;
using namespace cv;
Mat Nearest_neighbor_interpolation(const Mat&, int, int);
Mat binarization(Mat);
Mat grayImage(Mat);
Mat Bilinear_interpolation(Mat, int, int);
int main(){
    // link opencv
    #pragma comment(lib, "D:/OpenCV/build/x64/vc15/lib/opencv_world341d.lib")

    // read image
    Mat original = imread("images/lenna.png");

    // grayImage
    Mat gray = grayImage(original);

    // binarization
    binarization(gray);


    // Nearest neighbor interpolation implementation
    Nearest_neighbor_interpolation(original, 1200, 1200);

    // Bilinear_interpolation

    Bilinear_interpolation(original, 1200, 1200);

    waitKey(0);
    destroyAllWindows();

}

Mat Nearest_neighbor_interpolation(const Mat& image, int height, int width){
    // The shape of the initialization matrix, the matrix is equal to the target
    Mat result = cv::Mat::zeros(height, width, CV_8UC3);

    // To obtain scale than
    double scale_rate_height = (double)height / image.rows;
    double scale_rate_width = (double)width / image.cols;

    cout << "scale rate for height is " << scale_rate_height << endl;
    cout << "scale rate for width is " << scale_rate_width << endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            // get the current point
            auto &ptr = result.at<Vec3b>(i, j);
            // After get the original image to calculate the points
            int x = floor(i / scale_rate_height);
            int y = floor(j / scale_rate_width);
            auto &original = image.at<Vec3b>(x, y);
            // The assignment
            ptr[0] = original[0];
            ptr[1] = original[1];
            ptr[2] = original[2];
        }
    }
    imshow("result", result);
    return result;
}


Mat grayImage(Mat image){
    Mat empty = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    for (int i = 0; i < image.rows; ++i) {
        uchar *ptr = image.ptr(i);
        uchar *emp = empty.ptr(i);
        for (int j = 0; j < image.cols; ++j) {
            uchar *inner = ptr;
            emp[j] = (uchar)((uchar)inner[0] * 0.11 +(uchar)inner[1] * 0.59 + (uchar)inner[2] * 0.3);
            ptr += 3;
        }
    }
    imshow("gray", empty);
    return empty;
}


Mat binarization(Mat image){
    // initialization a matrix
    Mat empty = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);

    // Set the threshold
    int threshold = 140;
    for (int i = 0; i < image.rows; ++i) {
        uchar *ptr = image.ptr(i);
        uchar *emp = empty.ptr(i);
        for (int j = 0; j < image.cols; ++j) {
            if(ptr[j] > threshold){
                emp[j] = 255;
            }
            else{
                emp[j] = 0;
            }
        }
    }
    imshow("binarization", empty);
    return empty;
}

int is_a_negative_number(int i){
    if(i < 0){
        return 0;
    }
    return i;
}
Mat Bilinear_interpolation(Mat image, int height, int width){
    // initializer matrix
    Mat empty = cv::Mat::zeros(height, width, CV_8UC3);
    cout << "bilinear empty matrix dimension is " << empty.channels() << endl;

    // get scale rate
    double destination_rate_h = image.rows / (double)height;
    double destination_rate_w = image.cols / (double)width;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            auto &ptr = empty.at<Vec3b>(i, j);

            double src_x = (j + 0.5) * destination_rate_w - 0.5;
            double src_y = (i + 0.5) * destination_rate_h - 0.5;

            int src_x0 = (int)floor(src_x);
            int src_x1 = min(src_x0 + 1, image.cols - 1);

            int src_y0 = (int) floor(src_y);
            int src_y1 = min(src_y0 + 1, image.rows - 1);

            auto &originx1 = image.at<Vec3b>(is_a_negative_number(src_y0), is_a_negative_number(src_x0));
            auto &originx2 = image.at<Vec3b>(is_a_negative_number(src_y0), is_a_negative_number(src_x1));

            auto &originy1 = image.at<Vec3b>(is_a_negative_number(src_y1), is_a_negative_number(src_x0));
            auto &originy2 = image.at<Vec3b>(is_a_negative_number(src_y1), is_a_negative_number(src_x1));
            for (int k = 0; k < empty.channels(); k++) {
                double temp0 = (src_x1 - src_x) * originx1[k] + (src_x - src_x0) * originx2[k];
                double temp1 = (src_x1 - src_x) * originy1[k] + (src_x - src_x0) * originy2[k];
                ptr[k] = (int)((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1);
            }
        }
    }
    imshow("bilinear", empty);
    return empty;
}
