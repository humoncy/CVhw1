//#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <math.h>

using namespace std;
using namespace cv;

Mat Image, Image2, Image3, Image4, Image5, Image6;

Mat S = Mat(6, 3, CV_32F);


void ReadImages()
{
	Image = imread("test/bunny/pic1.bmp", IMREAD_GRAYSCALE);
	Image2 = imread("test/bunny/pic2.bmp", IMREAD_GRAYSCALE);
	Image3 = imread("test/bunny/pic3.bmp", IMREAD_GRAYSCALE);
	Image4 = imread("test/bunny/pic4.bmp", IMREAD_GRAYSCALE);
	Image5 = imread("test/bunny/pic5.bmp", IMREAD_GRAYSCALE);
	Image6 = imread("test/bunny/pic6.bmp", IMREAD_GRAYSCALE);
}

void ReadLightSources()
{
	fstream fin;
	string line;
	int a;
	fin.open("test/bunny/LightSource.txt", ios::in);
	if (!fin) {
		cout << "Fail to open file." << endl;
		return;
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
}

int main() {
	

	ReadImages();

	ReadLightSources();

	Mat NormalImage(Image.rows, Image.cols, CV_32FC3);
	Mat X_gradient = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat Y_gradient = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat Z_approx = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));

	//cout << Image.cols;

	//cout << S << endl;
	//cout << Z_approx.at<float>(0, 0) << endl;

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

			//cout << "I(x,y):\n" << I << endl;

			Mat B = ((S.t() * S).inv() * S.t() * I);
			//cout << "B(x,y):\n" << B << endl;
			float norm_b = sqrt(pow(B.at<float>(0, 0), 2.0) + pow(B.at<float>(1, 0), 2.0) + pow(B.at<float>(2, 0), 2.0));
			//cout << B.at<float>(2, 0);
			//cout << "norm_b:\n" << norm_b << endl;

			NormalImage.at<Vec3f>(rowIndex, colIndex)[0] = (norm_b == 0) ? B.at<float>(0, 0) : B.at<float>(0, 0) / norm_b;
			NormalImage.at<Vec3f>(rowIndex, colIndex)[1] = (norm_b == 0) ? B.at<float>(1, 0) : B.at<float>(1, 0) / norm_b;
			NormalImage.at<Vec3f>(rowIndex, colIndex)[2] = (norm_b == 0) ? B.at<float>(2, 0) : B.at<float>(2, 0) / norm_b;

			//cout << "N(x,y):\n" << NormalImage.at<Vec3f>(rowIndex, colIndex) << endl;

					
			float n1 = NormalImage.at<Vec3f>(rowIndex, colIndex)[0];
			float n2 = NormalImage.at<Vec3f>(rowIndex, colIndex)[1];
			float n3 = NormalImage.at<Vec3f>(rowIndex, colIndex)[2];
			
			//cout << "norm_N: " << pow(n1,2) + pow(n2,2) + pow(n3,2) << endl;
			X_gradient.at<float>(rowIndex, colIndex) = (n3 == 0) ? -n1 : -n1 / n3;
			Y_gradient.at<float>(rowIndex, colIndex) = (n3 == 0) ? -n2 : -n2 / n3;
			//Z_approx.at<float>(rowIndex, colIndex) = 4 * X_gradient.at<float>(rowIndex, colIndex) + Y_gradient.at<float>(rowIndex, colIndex);
			//cout << "Zapprox:\n" << Z_approx.at<float>(rowIndex, colIndex) << endl;
			//break;
		}
		//break;
	}
	
	//cout << Z_approx.at<float>(0, 0) << endl;

	for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
		for (int colIndex = 1; colIndex < Image.cols; colIndex++) {
			Z_approx.at<float>(rowIndex, colIndex) = Z_approx.at<float>(rowIndex, colIndex - 1) + X_gradient.at<float>(rowIndex, colIndex);
		}
	}
	for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
		for (int rowIndex = 1; rowIndex < Image.rows; rowIndex++) {
			Z_approx.at<float>(rowIndex, colIndex) = Z_approx.at<float>(rowIndex - 1, colIndex) + Y_gradient.at<float>(rowIndex, colIndex);
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
			fout << rowIndex << ' ' << colIndex << ' ' << Z_approx.at<float>(rowIndex, colIndex) << " 255 255 255" << endl;
			//fout << rowIndex << ' ' << colIndex << ' ' << "0.0" << " 255 255 255" << endl;
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



