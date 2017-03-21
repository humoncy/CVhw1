//#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <math.h>

using namespace std;
using namespace cv;

int main() {
	Mat Image = imread("test/bunny/pic1.bmp", IMREAD_GRAYSCALE);
	Mat Image2 = imread("test/bunny/pic2.bmp", IMREAD_GRAYSCALE);
	Mat Image3 = imread("test/bunny/pic3.bmp", IMREAD_GRAYSCALE);
	Mat Image4 = imread("test/bunny/pic4.bmp", IMREAD_GRAYSCALE);
	Mat Image5 = imread("test/bunny/pic5.bmp", IMREAD_GRAYSCALE);
	Mat Image6 = imread("test/bunny/pic6.bmp", IMREAD_GRAYSCALE);

	Mat normalImage(Image.rows, Image.cols, CV_32FC3);
	Mat S = Mat(6, 3, CV_32F);
	Mat Z_gradient = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat Z = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));

	fstream fin;
	string line;
	int a;
	std::string::size_type sz;   // alias of size_t
	fin.open("test/bunny/LightSource.txt", ios::in);
	if (!fin) {
		cout << "Fail to open file." << endl;
		return 0;
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
	for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
		for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
			Mat I = Mat(6, 1, CV_32F);
			I.at<float>(0, 0) = (float)Image.at<uchar>(rowIndex, colIndex);
			I.at<float>(1, 0) = (float)Image2.at<uchar>(rowIndex, colIndex);
			I.at<float>(2, 0) = (float)Image3.at<uchar>(rowIndex, colIndex);
			I.at<float>(3, 0) = (float)Image4.at<uchar>(rowIndex, colIndex);
			I.at<float>(4, 0) = (float)Image5.at<uchar>(rowIndex, colIndex);
			I.at<float>(5, 0) = (float)Image6.at<uchar>(rowIndex, colIndex);

			Mat B = ((S.t() * S).inv() * S.t() * I);
			//cout << B << endl;
			float norm_b = sqrt(pow(B.at<float>(0, 0), 2.0) + pow(B.at<float>(1, 0), 2.0) + pow(B.at<float>(2, 0), 2.0));
			//cout << B.at<float>(2, 0);
			//cout << norm_b;
			normalImage.at<Vec3f>(rowIndex, colIndex)[0] = B.at<float>(0, 0) / norm_b;
			normalImage.at<Vec3f>(rowIndex, colIndex)[1] = B.at<float>(1, 0) / norm_b;
			normalImage.at<Vec3f>(rowIndex, colIndex)[2] = B.at<float>(2, 0) / norm_b;
			//cout << normalImage.at<Vec3f>(rowIndex, colIndex) << endl;
			float n1 = normalImage.at<Vec3f>(rowIndex, colIndex)[0];
			float n2 = normalImage.at<Vec3f>(rowIndex, colIndex)[1];
			float n3 = normalImage.at<Vec3f>(rowIndex, colIndex)[2];
			float constant = 0.0;
			Z_gradient.at<float>(rowIndex, colIndex) = (-n1 / n3) * rowIndex + (-n2 / n3) * colIndex + constant;
		}
	}

	for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
		for (int colIndex = 1; colIndex < Image.cols; colIndex++) {
			Z.at<float>(rowIndex, colIndex) = Z.at<float>(rowIndex, colIndex - 1) + Z_gradient.at<float>(rowIndex, colIndex);
		}
	}
	for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
		for (int rowIndex = 1; rowIndex < Image.rows; rowIndex++) {
			Z.at<float>(rowIndex, colIndex) = Z.at<float>(rowIndex - 1, colIndex) + Z_gradient.at<float>(rowIndex, colIndex);
		}
	}

	fstream fout;
	fout.open("test/bunny/bunny_surface.ply", ios::out);
	if (fout.fail()) {
		cout << "Fail to open file." << endl;
		return 0;
	}

	fout << "ply\n";
	fout << "format ascii 1.0\n";
	fout << "comment alpha=1.0\n";
	fout << "element vertex 14400\n";
	fout << "property float x\n";
	fout << "property float y\n";
	fout << "property float z\n";
	fout << "property uchar red\n";
	fout << "property uchar green\n";
	fout << "property uchar blue z\n";
	fout << "end_header\n";

	for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
		for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
			fout << rowIndex << ' ' << colIndex << ' ' << Z.at<float>(rowIndex, colIndex) << " 255 255 255" << endl;
		}
	}

	//Mat result(Image.rows, Image.cols, CV_8U, Scalar(0));
	//// Rect ( x, y, width, height )
	//Image.copyTo(result(Rect(0, 0, tempImage.cols, tempImage.rows)));

	//imshow("CV", Image);
	//
	//waitKey();
	

	system("pause");
	return 0;
}



