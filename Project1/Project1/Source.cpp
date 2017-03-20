//#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int main() {
	Mat Image = imread("test/bunny/pic1.bmp", IMREAD_GRAYSCALE);
	Mat Image2 = imread("test/bunny/pic2.bmp", IMREAD_GRAYSCALE);
	Mat Image3 = imread("test/bunny/pic3.bmp", IMREAD_GRAYSCALE);
	Mat Image4 = imread("test/bunny/pic4.bmp", IMREAD_GRAYSCALE);
	Mat Image5 = imread("test/bunny/pic5.bmp", IMREAD_GRAYSCALE);
	Mat Image6 = imread("test/bunny/pic6.bmp", IMREAD_GRAYSCALE);

	Mat tempImage = Image.clone();
	Mat normalImage(Image.rows, Image.cols, CV_32FC3);
	Mat S = Mat(6, 3, CV_32F);

	fstream fin;
	string line;
	int a;
	std::string::size_type sz;   // alias of size_t
	fin.open("test/bunny/LightSource.txt", ios::in);
	if (!fin) {
		cout << "Fail to open file: " << endl;
	}
	// Read LightSource and initialize S
	for (int x = 0; x < S.rows; x++) {
		fin >> line;
		char a;
		fin >> a;
		for (int y = 0; y < S.cols; y++) {
			float b;
			fin >> b;
			//cout << b << ' ';
			S.at<float>(x, y) = b;
			fin >> a;
		}
	}

	// normal matrix
	for (int rowIndex = 100; rowIndex < Image.rows; rowIndex++) {
		for (int colIndex = 95; colIndex < Image.cols; colIndex++) {
			Mat I = Mat(6, 1, CV_32F);
			I.at<float>(0, 0) = (float)Image.at<uchar>(rowIndex, colIndex);
			I.at<float>(1, 0) = (float)Image2.at<uchar>(rowIndex, colIndex);
			I.at<float>(2, 0) = (float)Image3.at<uchar>(rowIndex, colIndex);
			I.at<float>(3, 0) = (float)Image4.at<uchar>(rowIndex, colIndex);
			I.at<float>(4, 0) = (float)Image5.at<uchar>(rowIndex, colIndex);
			I.at<float>(5, 0) = (float)Image6.at<uchar>(rowIndex, colIndex);

			Mat B = ((S.t() * S).inv() * S.t() * I);
			cout << B << endl;
			//cout << B.at<float>(2, 0);
			normalImage.at<Vec3f>(rowIndex, colIndex)[0] = B.at<float>(0, 0);
			normalImage.at<Vec3f>(rowIndex, colIndex)[1] = B.at<float>(1, 0);
			normalImage.at<Vec3f>(rowIndex, colIndex)[2] = B.at<float>(2, 0);

			//cout << normalImage.at<Vec3f>(rowIndex, colIndex) << endl;

			break;
		}
		break;
	}

	//Mat result(Image.rows, Image.cols, CV_8U, Scalar(0));
	//// Rect ( x, y, width, height )
	//Image.copyTo(result(Rect(0, 0, tempImage.cols, tempImage.rows)));

	//imshow("CV", result);
	//
	//waitKey();
	

	system("pause");
	return 0;
}



