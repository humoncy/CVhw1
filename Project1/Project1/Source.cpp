//#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

Mat Image, Image2, Image3, Image4, Image5, Image6;

Mat S = Mat(6, 3, CV_32F);

float norm(float a, float b, float c)
{
	return sqrt(pow(a, 2.0) + pow(b, 2.0) + pow(c, 2.0));
}

void readImages()
{
	Image = imread("test/bunny/pic1.bmp", IMREAD_GRAYSCALE);
	Image2 = imread("test/bunny/pic2.bmp", IMREAD_GRAYSCALE);
	Image3 = imread("test/bunny/pic3.bmp", IMREAD_GRAYSCALE);
	Image4 = imread("test/bunny/pic4.bmp", IMREAD_GRAYSCALE);
	Image5 = imread("test/bunny/pic5.bmp", IMREAD_GRAYSCALE);
	Image6 = imread("test/bunny/pic6.bmp", IMREAD_GRAYSCALE);
}

void readLightSources()
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
		int norm_S = norm(S.at<float>(x, 0), S.at<float>(x, 1), S.at<float>(x, 2));
		S.at<float>(x, 0) /= norm_S;
		S.at<float>(x, 1) /= norm_S;
		S.at<float>(x, 2) /= norm_S;
	}
}

void writePly(Mat Z_approx)
{
	fstream fout;
	fout.open("test/bunny/bunny_surface.ply", ios::out);
	if (fout.fail()) {
		cout << "Fail to open file." << endl;
		return;
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
}

void computeNormalandGradient(Mat NormalImage,Mat X_gradient,Mat Y_gradient)
{
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
			float norm_b = norm(B.at<float>(0, 0), B.at<float>(1, 0), B.at<float>(2, 0));
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
			X_gradient.at<float>(rowIndex, colIndex) = (n3 == 0) ? n1 : n1 / n3;
			Y_gradient.at<float>(rowIndex, colIndex) = (n3 == 0) ? n2 : n2 / n3;
		}
	}
	
}

void integral(Mat dst, Mat X_gradient, Mat Y_gradient, int start_x, int start_y, int direction_x, int direction_y, bool order)
{
	if (order) {
		for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
			int colIndex = start_x + direction_x;
			for (int i = 1; i < Image.cols; i++) {
				
				dst.at<float>(rowIndex, colIndex) = dst.at<float>(rowIndex, colIndex - direction_x) + direction_x * X_gradient.at<float>(rowIndex, colIndex);
				colIndex += direction_x;
			}
		}
		for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
			int rowIndex = start_y + direction_y;
			for (int i = 1; i < Image.cols; i++) {
				dst.at<float>(rowIndex, colIndex) = dst.at<float>(rowIndex - direction_y, colIndex) + direction_y * Y_gradient.at<float>(rowIndex, colIndex);
				rowIndex += direction_y;
			}
		}
	}
	else {
		for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
			int rowIndex = start_y + direction_y;
			for (int i = 1; i < Image.cols; i++) {
				dst.at<float>(rowIndex, colIndex) = dst.at<float>(rowIndex - direction_y, colIndex) - direction_y * Y_gradient.at<float>(rowIndex, colIndex);
				rowIndex += direction_y;
			}
		}
		for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
			int colIndex = start_x + direction_x;
			for (int i = 1; i < Image.cols; i++) {
				dst.at<float>(rowIndex, colIndex) = dst.at<float>(rowIndex, colIndex - direction_x) - direction_x * X_gradient.at<float>(rowIndex, colIndex);
				colIndex += direction_x;
			}
		}
	}
}

void surface_Reconstruction_Integration(Mat Z_approx, Mat X_gradient, Mat Y_gradient, Mat NormalImage)
{
	Mat integral_RD = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat integral_DR = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat integral_RU = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat integral_UR = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat integral_LD = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat integral_DL = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat integral_LU = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat integral_UL = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));

	integral(integral_RD, X_gradient, Y_gradient, 0, 0, 1, 1, 1);
	integral(integral_RU, X_gradient, Y_gradient, 0, 119, 1, -1, 1);
	integral(integral_LD, X_gradient, Y_gradient, 119, 0, -1, 1, 1);
	integral(integral_LU, X_gradient, Y_gradient, 119, 119, -1, -1, 1);
	integral(integral_DR, X_gradient, Y_gradient, 0, 0, 1, 1, 0);
	integral(integral_UR, X_gradient, Y_gradient, 0, 119, 1, -1, 0);
	integral(integral_DL, X_gradient, Y_gradient, 119, 0, -1, 1, 0);
	integral(integral_UL, X_gradient, Y_gradient, 119, 119, -1, -1, 0);

	for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
		for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
			float RD = integral_RD.at<float>(rowIndex, colIndex);
			float RU = integral_RU.at<float>(rowIndex, colIndex);
			float LD = integral_LD.at<float>(rowIndex, colIndex);
			float LU = integral_LU.at<float>(rowIndex, colIndex);
			float DR = integral_DR.at<float>(rowIndex, colIndex);
			float DL = integral_DL.at<float>(rowIndex, colIndex);
			float UR = integral_UR.at<float>(rowIndex, colIndex);
			float UL = integral_UL.at<float>(rowIndex, colIndex);

			float tmp[] = {RD, RU, LD, LU, DR, DL, UR, UL};
			vector<float> integrations(tmp, tmp+8);

			//sort(integrations.begin(), integrations.end());
			
			/*std::cout << "integrations contains:";
			for (std::vector<float>::iterator it = integrations.begin(); it != integrations.end(); ++it)
				std::cout << ' ' << *it;
			std::cout << '\n';*/

			//int cnt = count_if(integrations.begin(), integrations.end(), [](int i) {return i == 0; });
			//cout << cnt << endl;
			float normal_length = norm(NormalImage.at<Vec3f>(rowIndex, colIndex)[0], NormalImage.at<Vec3f>(rowIndex, colIndex)[1], NormalImage.at<Vec3f>(rowIndex, colIndex)[2]);
			if ( normal_length == 0.0 ) {
				Z_approx.at<float>(rowIndex, colIndex) = 0.0;
			}
			else {
				//Z_approx.at<float>(rowIndex, colIndex) = LD;

				Mat tmp_m, tmp_sd;
				meanStdDev(integrations, tmp_m, tmp_sd);

				double mean = tmp_m.at<double>(0, 0);
				double sd = tmp_sd.at<double>(0, 0);
				/*cout << "mean: " << mean << endl;
				cout << "std: " << sd << endl;*/

				int count_good = 0;
				for (std::vector<float>::iterator it = integrations.begin(); it != integrations.end(); ++it) {
					if ( abs(*it - mean) <= sd ) {
						Z_approx.at<float>(rowIndex, colIndex) += *it;
						++count_good;
					}
				}
				Z_approx.at<float>(rowIndex, colIndex) /= count_good;

				/*Z_approx.at<float>(rowIndex, colIndex) = (
					integral_RD.at<float>(rowIndex, colIndex) + integral_RU.at<float>(rowIndex, colIndex) +
					integral_LD.at<float>(rowIndex, colIndex) + integral_LU.at<float>(rowIndex, colIndex) +
					integral_DR.at<float>(rowIndex, colIndex) + integral_DL.at<float>(rowIndex, colIndex) +
					integral_UR.at<float>(rowIndex, colIndex) + integral_UL.at<float>(rowIndex, colIndex)) / 8;*/
			}
			
		}
	}
	
}

int main() {
	

	readImages();

	readLightSources();

	Mat NormalImage(Image.rows, Image.cols, CV_32FC3);
	Mat X_gradient = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat Y_gradient = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat Z_approx = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));


	//cout << Image.cols;

	//cout << S << endl;
	//cout << Z_approx.at<float>(0, 0) << endl;

	computeNormalandGradient(NormalImage, X_gradient, Y_gradient);	
	
	//cout << Z_approx.at<float>(0, 0) << endl;

	surface_Reconstruction_Integration(Z_approx, X_gradient, Y_gradient, NormalImage);

	Mat Z = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Z_approx.copyTo(Z(Rect(0, 0, Image.cols, Image.rows)));
	//GaussianBlur(Z, Z, Size(9, 9), 0, 0);
	GaussianBlur(Z, Z, Size(3, 3), 0, 0);
	Laplacian(Z, Z_approx, CV_32F, 3, 1, 0, BORDER_DEFAULT);
	Z_approx = Z + Z_approx;
	cv::bilateralFilter(Z_approx, Z, 5, 3, 3);

	//medianBlur(Z_approx, Z, 5);

	writePly(Z);

	//Mat result(Image.rows, Image.cols, CV_8U, Scalar(0));
	//// Rect ( x, y, width, height )
	//Image.copyTo(result(Rect(0, 0, tempImage.cols, tempImage.rows)));

	Mat try_fun(Image.rows, Image.cols, CV_32FC3);
	NormalImage.copyTo(try_fun(Rect(0, 0, NormalImage.cols, NormalImage.rows)));
	//bilateralFilter(NormalImage, try_fun, 15, 80, 80);
	imshow("CV", try_fun);
	//
	waitKey();
	

	//system("pause");
	return 0;
}



