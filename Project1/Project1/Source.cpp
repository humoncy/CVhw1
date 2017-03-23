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
}

void surface_Reconstruction_Integration(Mat Z_approx, Mat X_gradient, Mat Y_gradient)
{
	Mat X_integral_LtoR = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat X_integral_RtoL = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat Y_integral_UtoD = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat Y_integral_DtoU = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));


	for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
		for (int colIndex = 1; colIndex < Image.cols; colIndex++) {
			int colIndex_inv = Image.cols - colIndex - 1; // 118~0
			X_integral_LtoR.at<float>(rowIndex, colIndex) = X_integral_LtoR.at<float>(rowIndex, colIndex - 1) + X_gradient.at<float>(rowIndex, colIndex);
			X_integral_RtoL.at<float>(rowIndex, colIndex_inv) = X_integral_RtoL.at<float>(rowIndex, colIndex_inv + 1) + X_gradient.at<float>(rowIndex, colIndex_inv);

			//Z_approx.at<float>(rowIndex, colIndex) = Z_approx.at<float>(rowIndex, colIndex - 1) + X_gradient.at<float>(rowIndex, colIndex);
		}
	}
	for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
		for (int rowIndex = 1; rowIndex < Image.rows; rowIndex++) {
			int rowIndex_inv = Image.rows - rowIndex - 1; // 118~0
			Y_integral_UtoD.at<float>(rowIndex, colIndex) = Y_integral_UtoD.at<float>(rowIndex - 1, colIndex) + Y_gradient.at<float>(rowIndex, colIndex);
			Y_integral_DtoU.at<float>(rowIndex, rowIndex_inv) = Y_integral_DtoU.at<float>(rowIndex_inv + 1, colIndex) + Y_gradient.at<float>(rowIndex_inv, colIndex);

			//Z_approx.at<float>(rowIndex, colIndex) = Z_approx.at<float>(rowIndex - 1, colIndex) + Y_gradient.at<float>(rowIndex, colIndex);
		}
	}

	for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
		for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
			int colIndex_inv = Image.cols - colIndex - 1; // 118~0
			int w_LtoR = Image.cols - colIndex;  // weight from left
			int w_RtoL = colIndex;
			Z_approx.at<float>(rowIndex, colIndex) = (X_integral_LtoR.at<float>(rowIndex, colIndex) * w_LtoR + X_integral_RtoL.at<float>(rowIndex, colIndex_inv) * w_RtoL) / Image.cols;

		}
	}

	for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
		for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
			int rowIndex_inv = Image.rows - rowIndex - 1; // 118~0
			int w_UtoD = Image.rows - rowIndex;  // weight from left
			int w_DtoU = rowIndex;
			Z_approx.at<float>(rowIndex, colIndex) += (Y_integral_UtoD.at<float>(rowIndex, colIndex) * w_UtoD + Y_integral_DtoU.at<float>(rowIndex_inv, colIndex) * w_DtoU) / Image.rows;

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

	surface_Reconstruction_Integration(Z_approx, X_gradient, Y_gradient);

	

	// Z = 0 when normal is zero
	for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
		for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
			float normal_length = norm(NormalImage.at<Vec3f>(rowIndex, colIndex)[0], NormalImage.at<Vec3f>(rowIndex, colIndex)[1], NormalImage.at<Vec3f>(rowIndex, colIndex)[2]);
			if ( normal_length == 0.0 ) {
				Z_approx.at<float>(rowIndex, colIndex) = 0.0;
			}
		}
	}

	writePly(Z_approx);

	//Mat result(Image.rows, Image.cols, CV_8U, Scalar(0));
	//// Rect ( x, y, width, height )
	//Image.copyTo(result(Rect(0, 0, tempImage.cols, tempImage.rows)));

	//imshow("CV", NormalImage);
	//
	//waitKey();
	

	system("pause");
	return 0;
}



