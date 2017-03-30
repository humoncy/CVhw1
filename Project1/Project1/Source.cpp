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
Mat NormalImage;

/*************************************************************************/
/* ========== Change the parameters here to see what you want  ==========*/
bool special = 0;
string folder = "venus";
/* ======================================================================*/
/*************************************************************************/

string filename = folder + "_surface";

Mat S = Mat(6, 3, CV_32F);

float norm(float a, float b, float c)
{
	return sqrt(pow(a, 2.0) + pow(b, 2.0) + pow(c, 2.0));
}

void readImages()
{
	cout << folder << endl;
	if (!special) {
		cout << "Not special case." << endl;
		Image = imread("test/" + folder + "/pic1.bmp", IMREAD_GRAYSCALE);
		Image2 = imread("test/" + folder + "/pic2.bmp", IMREAD_GRAYSCALE);
		Image3 = imread("test/" + folder + "/pic3.bmp", IMREAD_GRAYSCALE);
		Image4 = imread("test/" + folder + "/pic4.bmp", IMREAD_GRAYSCALE);
		Image5 = imread("test/" + folder + "/pic5.bmp", IMREAD_GRAYSCALE);
		Image6 = imread("test/" + folder + "/pic6.bmp", IMREAD_GRAYSCALE);
	}
	else {
		cout << "Special case" << endl;
		Image = imread("test/special/" + folder + "/pic1.bmp", IMREAD_GRAYSCALE);
		Image2 = imread("test/special/" + folder + "/pic2.bmp", IMREAD_GRAYSCALE);
		Image3 = imread("test/special/" + folder + "/pic3.bmp", IMREAD_GRAYSCALE);
		Image4 = imread("test/special/" + folder + "/pic4.bmp", IMREAD_GRAYSCALE);
		Image5 = imread("test/special/" + folder + "/pic5.bmp", IMREAD_GRAYSCALE);
		Image6 = imread("test/special/" + folder + "/pic6.bmp", IMREAD_GRAYSCALE);
		medianBlur(Image, Image, 3);
		medianBlur(Image2, Image2, 3);
		medianBlur(Image3, Image3, 3);
		medianBlur(Image4, Image4, 3);
		medianBlur(Image5, Image5, 3);
		medianBlur(Image6, Image6, 3);
	}
}

void readLightSources()
{
	fstream fin;
	string line;
	if (!special) {
		fin.open("test/" + folder + "/LightSource.txt", ios::in);
	}
	else {
		fin.open("test/special/" + folder + "/LightSource.txt", ios::in);
	}
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
			S.at<float>(x, y) = b;
			fin >> a;
		}
		int norm_S = norm(S.at<float>(x, 0), S.at<float>(x, 1), S.at<float>(x, 2));
		S.at<float>(x, 0) /= norm_S;
		S.at<float>(x, 1) /= norm_S;
		S.at<float>(x, 2) /= norm_S;
	}
}

void writePly(Mat Z_approx, string filename)
{
	fstream fout;
	if (!special) {
		fout.open("test/" + folder + "/" + filename + ".ply", ios::out);
	}
	else {
		fout.open("test/special/" + folder + "/" + filename + ".ply", ios::out);
	}
	if (fout.fail()) {
		cout << "Fail to open file." << endl;
		return;
	}

	fout << "ply\n";
	fout << "format ascii 1.0\n";
	fout << "comment alpha=1.0\n";
	fout << "element vertex " << Image.rows * Image.cols << endl;
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
		}
	}
}

void makeWeightedMatrix(Mat W, Mat I)
{
	for (int i = 0; i < 6; i++) {
		float tmp_Ixy = I.at<float>(i, 0);
		if (tmp_Ixy > 250 || tmp_Ixy < 10) {
			W.at<float>(i, i) = 0.001;
		}
		else {
			W.at<float>(i, i) = 1;
		}
	}
}

void computeNormalandGradient(Mat NormalImage, Mat X_gradient, Mat Y_gradient)
{
	for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
		for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
			Mat I = Mat(6, 1, CV_32F);
			Mat W = Mat::eye(6, 6, CV_32F);

			I.at<float>(0, 0) = (float)Image.at<uchar>(rowIndex, colIndex);
			I.at<float>(1, 0) = (float)Image2.at<uchar>(rowIndex, colIndex);
			I.at<float>(2, 0) = (float)Image3.at<uchar>(rowIndex, colIndex);
			I.at<float>(3, 0) = (float)Image4.at<uchar>(rowIndex, colIndex);
			I.at<float>(4, 0) = (float)Image5.at<uchar>(rowIndex, colIndex);
			I.at<float>(5, 0) = (float)Image6.at<uchar>(rowIndex, colIndex);

			makeWeightedMatrix(W, I);

			Mat WS = Mat(6, 1, CV_32F);
			WS = W * S;

			Mat B = (WS.t() * WS).inv() * WS.t() * W * I;
			//Mat B = (S.t() * S).inv() * S.t() * I;

			float norm_b = norm(B.at<float>(0, 0), B.at<float>(1, 0), B.at<float>(2, 0));

			NormalImage.at<Vec3f>(rowIndex, colIndex)[0] = (norm_b == 0) ? B.at<float>(0, 0) : B.at<float>(0, 0) / norm_b;
			NormalImage.at<Vec3f>(rowIndex, colIndex)[1] = (norm_b == 0) ? B.at<float>(1, 0) : B.at<float>(1, 0) / norm_b;
			NormalImage.at<Vec3f>(rowIndex, colIndex)[2] = (norm_b == 0) ? B.at<float>(2, 0) : B.at<float>(2, 0) / norm_b;

			float n1 = NormalImage.at<Vec3f>(rowIndex, colIndex)[0];
			float n2 = NormalImage.at<Vec3f>(rowIndex, colIndex)[1];
			float n3 = NormalImage.at<Vec3f>(rowIndex, colIndex)[2];

			X_gradient.at<float>(rowIndex, colIndex) = (n3 == 0) ? n2 : n2 / n3;
			Y_gradient.at<float>(rowIndex, colIndex) = (n3 == 0) ? -n1 : -n1 / n3;
		}
	}
	medianBlur(X_gradient, X_gradient, 5);
	medianBlur(Y_gradient, Y_gradient, 5);
}

// This is the first version of my integration, just keep it as memory. I don't use this version for my integration
// order is true : first horizontal then vertical
//          false: 
void integral(Mat dst, Mat X_gradient, Mat Y_gradient, int start_x, int start_y, int direction_x, int direction_y, bool order)
{
	if (order) {
		for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
			int colIndex = start_y + direction_x;
			for (int i = 1; i < Image.cols; i++) {
				dst.at<float>(rowIndex, colIndex) = dst.at<float>(rowIndex, colIndex - direction_x) + direction_x * X_gradient.at<float>(rowIndex, colIndex);
				colIndex += direction_x;
			}
		}
		for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
			int rowIndex = start_x + direction_y;
			for (int i = 1; i < Image.cols; i++) {
				dst.at<float>(rowIndex, colIndex) = dst.at<float>(rowIndex - direction_y, colIndex) + direction_y * Y_gradient.at<float>(rowIndex, colIndex);
				rowIndex += direction_y;
			}
		}
	}
	else {
		for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
			int rowIndex = start_x + direction_y;
			for (int i = 1; i < Image.cols; i++) {
				dst.at<float>(rowIndex, colIndex) = dst.at<float>(rowIndex - direction_y, colIndex) - direction_y * Y_gradient.at<float>(rowIndex, colIndex);
				rowIndex += direction_y;
			}
		}
		for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
			int colIndex = start_y + direction_x;
			for (int i = 1; i < Image.cols; i++) {
				dst.at<float>(rowIndex, colIndex) = dst.at<float>(rowIndex, colIndex - direction_x) - direction_x * X_gradient.at<float>(rowIndex, colIndex);
				colIndex += direction_x;
			}
		}
	}
}
// order is true : first horizontal then vertical
//          false: first vertical then horizontal
// Because the gradient has directionality, when to use add or subtract matters
void integral_(Mat dst, Mat X_gradient, Mat Y_gradient, int start_x, int start_y, int direction_x, int direction_y, bool order)
{
	if (order) {
		int colIndex = start_y + direction_y;
		for (int i = 1; i < Image.cols; i++) {
			dst.at<float>(start_x, colIndex) = dst.at<float>(start_x, colIndex - direction_y) - direction_y * Y_gradient.at<float>(start_x, colIndex);
			colIndex += direction_y;
		}
		for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
			int rowIndex = start_x + direction_x;
			for (int i = 1; i < Image.cols; i++) {
				dst.at<float>(rowIndex, colIndex) = dst.at<float>(rowIndex - direction_x, colIndex) + direction_x * X_gradient.at<float>(rowIndex, colIndex);
				rowIndex += direction_x;
			}
		}
	}
	else {
		int rowIndex = start_x + direction_x;
		for (int i = 1; i < Image.cols; i++) {
			dst.at<float>(rowIndex, start_y) = dst.at<float>(rowIndex - direction_x, start_y) - direction_x * X_gradient.at<float>(rowIndex, start_y);
			rowIndex += direction_x;
		}
		for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
			int colIndex = start_y + direction_y;
			for (int i = 1; i < Image.cols; i++) {
				dst.at<float>(rowIndex, colIndex) = dst.at<float>(rowIndex, colIndex - direction_y) + direction_y * Y_gradient.at<float>(rowIndex, colIndex);
				colIndex += direction_y;
			}
		}
	}
}

// check whether the numerical 2nd order derivatives are close to each other
void sanity_check(Mat check_derivatives, Mat X_gradient, Mat Y_gradient)
{
	Mat derivative_xy = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat derivative_yx = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));

	//  2nd order derivatives, just like 1x3 fiter [-1 0 1] on gradients
	for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
		for (int colIndex = 1; colIndex < Image.cols - 1; colIndex++) {
			derivative_xy.at<float>(rowIndex, colIndex) = -X_gradient.at<float>(rowIndex, colIndex - 1) + X_gradient.at<float>(rowIndex, colIndex + 1);
		}
	}
	for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
		for (int rowIndex = 1; rowIndex < Image.rows - 1; rowIndex++) {
			derivative_yx.at<float>(rowIndex, colIndex) = -Y_gradient.at<float>(rowIndex - 1, colIndex) + Y_gradient.at<float>(rowIndex + 1, colIndex);
		}
	}
	imshow("X_gradient", X_gradient);
	imshow("Y_gradient", Y_gradient);

	for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
		for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
			float diff = derivative_xy.at<float>(rowIndex, colIndex) - derivative_yx.at<float>(rowIndex, colIndex);
			if (pow(diff, 2.0) <= 0.035) {
				check_derivatives.at<float>(rowIndex, colIndex) = 0.0;
			}
			// Let the gradient be not so representative.
			else {
				check_derivatives.at<float>(rowIndex, colIndex) = 1.0;
				float x_sum = 0.0, y_sum = 0.0;
				for (int x = -2; x <= 2; x++) {
					for (int y = -2; y <= 2; y++) {
						x_sum += X_gradient.at<float>(rowIndex + x, colIndex + y);
						y_sum += Y_gradient.at<float>(rowIndex + x, colIndex + y);
					}
				}
				X_gradient.at<float>(rowIndex, colIndex) = x_sum / 25;
				Y_gradient.at<float>(rowIndex, colIndex) = y_sum / 25;
			}
		}
	}
	imshow("sanity check", check_derivatives);
}

void surface_Reconstruction_Integration(Mat Z_approx, Mat X_gradient, Mat Y_gradient, Mat NormalImage)
{
	// R go right, L go left, D go down, U go up
	// RD means first go right then go down, just as below:
	//                  ------->
	//                         |
	//                         |
	//                         v
	// and so on.

	Mat integral_RD = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat integral_DR = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat integral_RU = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	///Mat integral_UR = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	///Mat integral_LD = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat integral_DL = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	///Mat integral_LU = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	///Mat integral_UL = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));

	/*** Integrathions from eight directions, but actually there are only 4 results, that's why I comment 4 of them ***/

	integral_(integral_RD, X_gradient, Y_gradient, 0, 0, 1, 1, 1);
	integral_(integral_RU, X_gradient, Y_gradient, Image.rows - 1, 0, -1, 1, 1);
	//integral_(integral_LD, X_gradient, Y_gradient, 0, Image.cols - 1, 1, -1, 1);
	//integral_(integral_LU, X_gradient, Y_gradient, Image.rows - 1, Image.cols - 1, -1, -1, 1);
	integral_(integral_DR, X_gradient, Y_gradient, 0, 0, 1, 1, 0);
	//integral_(integral_UR, X_gradient, Y_gradient, Image.rows - 1, 0, -1, 1, 0);
	integral_(integral_DL, X_gradient, Y_gradient, 0, Image.cols - 1, 1, -1, 0);
	//integral_(integral_UL, X_gradient, Y_gradient, Image.rows - 1, Image.cols - 1, -1, -1, 0);

	for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
		for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
			float RD = integral_RD.at<float>(rowIndex, colIndex);
			float RU = integral_RU.at<float>(rowIndex, colIndex);
			//float LD = integral_LD.at<float>(rowIndex, colIndex);
			//float LU = integral_LU.at<float>(rowIndex, colIndex);
			float DR = integral_DR.at<float>(rowIndex, colIndex);
			float DL = integral_DL.at<float>(rowIndex, colIndex);
			//float UR = integral_UR.at<float>(rowIndex, colIndex);
			//float UL = integral_UL.at<float>(rowIndex, colIndex);

			//float tmp[] = {RD, RU, LD, LU, DR, DL, UR, UL};
			//vector<float> integrations(tmp, tmp+8);
			float tmp[] = { RD, RU, DR, DL };
			vector<float> integrations(tmp, tmp + 4);

			//sort(integrations.begin(), integrations.end());

			/*std::cout << "integrations contains:";
			for (std::vector<float>::iterator it = integrations.begin(); it != integrations.end(); ++it)
			std::cout << ' ' << *it;
			std::cout << '\n';*/

			//int cnt = count_if(integrations.begin(), integrations.end(), [](int i) {return i == 0; });
			//cout << cnt << endl;
			float normal_length = norm(NormalImage.at<Vec3f>(rowIndex, colIndex)[0], NormalImage.at<Vec3f>(rowIndex, colIndex)[1], NormalImage.at<Vec3f>(rowIndex, colIndex)[2]);
			if (normal_length == 0.0) {
				Z_approx.at<float>(rowIndex, colIndex) = 0.0;
				integral_RD.at<float>(rowIndex, colIndex) = 0.0;
				integral_RU.at<float>(rowIndex, colIndex) = 0.0;
				//integral_LD.at<float>(rowIndex, colIndex) = 0.0;
				//integral_LU.at<float>(rowIndex, colIndex) = 0.0;
				integral_DR.at<float>(rowIndex, colIndex) = 0.0;
				integral_DL.at<float>(rowIndex, colIndex) = 0.0;
				//integral_UR.at<float>(rowIndex, colIndex) = 0.0;
				//integral_UL.at<float>(rowIndex, colIndex) = 0.0;
			}
			else {
				Mat tmp_m, tmp_sd;
				meanStdDev(integrations, tmp_m, tmp_sd);

				double mean = tmp_m.at<double>(0, 0);
				double sd = tmp_sd.at<double>(0, 0);
				/*cout << "mean: " << mean << endl;
				cout << "std: " << sd << endl;*/

				int count_good = 0;
				for (std::vector<float>::iterator it = integrations.begin(); it != integrations.end(); ++it) {
					if (abs(*it - mean) <= 2 * sd) {
						Z_approx.at<float>(rowIndex, colIndex) += *it;
						++count_good;
					}
				}
				Z_approx.at<float>(rowIndex, colIndex) /= count_good;

				/*Z_approx.at<float>(rowIndex, colIndex) = (
				integral_RD.at<float>(rowIndex, colIndex) + integral_RU.at<float>(rowIndex, colIndex) +
				integral_DR.at<float>(rowIndex, colIndex) + integral_DL.at<float>(rowIndex, colIndex)) / 4;*/
			}
		}
	}

	double min, max;
	cv::minMaxLoc(Z_approx, &min, &max);

	for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
		for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
			if (Z_approx.at<float>(rowIndex, colIndex) != 0.0) {
				Z_approx.at<float>(rowIndex, colIndex) = (Z_approx.at<float>(rowIndex, colIndex) - min);
			}
		}
	}

	GaussianBlur(Z_approx, Z_approx, Size(5, 5), 0, 0);


	/*writePly(integral_RD, "RD");
	writePly(integral_RU, "RU");
	writePly(integral_LD, "LD");
	writePly(integral_LU, "LU");
	writePly(integral_DR, "DR");
	writePly(integral_DL, "DL");
	writePly(integral_UR, "UR");
	writePly(integral_UL, "UL");*/
}

/* To compute (AU-B)^2 */
float jacobi_Error(Mat U, Mat B)
{
	float loss_sum = 0.0;
	for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
		for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
			if (rowIndex == 0 && colIndex == 0) {
				loss_sum += pow((-4 * U.at<float>(rowIndex*Image.cols + colIndex, 0) + U.at<float>(rowIndex*Image.cols + (colIndex + 1), 0) + U.at<float>((rowIndex + 1)*Image.cols + colIndex, 0)) - B.at<float>(rowIndex*Image.cols + colIndex, 0), 2.0);
			}
			else if (rowIndex == Image.rows - 1 && colIndex == Image.cols - 1) {
				loss_sum += pow((-4 * U.at<float>(rowIndex*Image.cols + colIndex, 0) + U.at<float>(rowIndex*Image.cols + (colIndex - 1), 0) + U.at<float>((rowIndex - 1)*Image.cols + colIndex, 0)) - B.at<float>(rowIndex*Image.cols + colIndex, 0), 2.0);
			}
			else if (rowIndex == 0 && colIndex != 0) {
				loss_sum += pow((-4 * U.at<float>(rowIndex*Image.cols + colIndex, 0) + U.at<float>(rowIndex*Image.cols + (colIndex + 1), 0) + U.at<float>((rowIndex + 1)*Image.cols + colIndex, 0) + U.at<float>(rowIndex*Image.cols + (colIndex - 1), 0)) - B.at<float>(rowIndex*Image.cols + colIndex, 0), 2.0);
			}
			else if (rowIndex == Image.rows - 1 && colIndex != Image.cols - 1) {
				loss_sum += pow((-4 * U.at<float>(rowIndex*Image.cols + colIndex, 0) + U.at<float>(rowIndex*Image.cols + (colIndex - 1), 0) + U.at<float>((rowIndex - 1)*Image.cols + colIndex, 0) + U.at<float>(rowIndex*Image.cols + (colIndex + 1), 0)) - B.at<float>(rowIndex*Image.cols + colIndex, 0), 2.0);
			}
			else {
				loss_sum += pow((-4 * U.at<float>(rowIndex*Image.cols + colIndex, 0) + U.at<float>(rowIndex*Image.cols + (colIndex + 1), 0) + U.at<float>((rowIndex + 1)*Image.cols + colIndex, 0) + U.at<float>(rowIndex*Image.cols + (colIndex - 1), 0) + U.at<float>((rowIndex - 1)*Image.cols + colIndex, 0)) - B.at<float>(rowIndex*Image.cols + colIndex, 0), 2.0);
			}
		}
	}
	return loss_sum;
}

/* Just laplacian [-1 0 1] */
void Laplacian_(Mat Laplacian_Z, Mat X_gradient, Mat Y_gradient, Mat check_derivatives)
{
	Mat derivative_xx = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat derivative_yy = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));

	for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
		for (int rowIndex = 1; rowIndex < Image.rows - 1; rowIndex++) {
			derivative_xx.at<float>(rowIndex, colIndex) = -X_gradient.at<float>(rowIndex - 1, colIndex) + X_gradient.at<float>(rowIndex + 1, colIndex);
		}
	}

	for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
		for (int colIndex = 1; colIndex < Image.cols - 1; colIndex++) {
			derivative_yy.at<float>(rowIndex, colIndex) = -Y_gradient.at<float>(rowIndex, colIndex - 1) + Y_gradient.at<float>(rowIndex, colIndex + 1);
		}
	}

	Laplacian_Z = derivative_xx + derivative_yy;
}

/* Another kind of Laplacian [-1 1] */
void Laplacian_v2(Mat Laplacian_Z, Mat X_gradient, Mat Y_gradient)
{
	Mat derivative_xx = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat derivative_yy = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));

	for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
		for (int rowIndex = 1; rowIndex < Image.rows; rowIndex++) {
			derivative_xx.at<float>(rowIndex, colIndex) = -X_gradient.at<float>(rowIndex - 1, colIndex) + X_gradient.at<float>(rowIndex, colIndex);
		}
	}

	for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
		for (int colIndex = 1; colIndex < Image.cols; colIndex++) {
			derivative_yy.at<float>(rowIndex, colIndex) = -Y_gradient.at<float>(rowIndex, colIndex - 1) + Y_gradient.at<float>(rowIndex, colIndex);
		}
	}


	Laplacian_Z = derivative_xx + derivative_yy;
}

void sharping(Mat Z_approx, Mat check_derivatives)
{
	Mat Z_filter = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	cv::bilateralFilter(Z_approx, Z_filter, 5, 100, 100);
	Mat G_mask = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	G_mask = Z_approx - Z_filter;
	imshow("unsharp mask", G_mask);
	for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
		for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
			float normal_length = norm(NormalImage.at<Vec3f>(rowIndex, colIndex)[0], NormalImage.at<Vec3f>(rowIndex, colIndex)[1], NormalImage.at<Vec3f>(rowIndex, colIndex)[2]);
			if (normal_length == 0.0) {
				G_mask.at<float>(rowIndex, colIndex) = 0.0;
			}
			if (check_derivatives.at<float>(rowIndex, colIndex) == 1.0) {
				G_mask.at<float>(rowIndex, colIndex) *= 1.5;
			}
		}
	}
	Z_approx += G_mask * 3;
}

/*******************************************************************/
// Concept: AU=B, then solve U
// To reconstruct surface:
// A: Laplacian
// U: Z, unknown
// B: Laplacian of Z, which is equal to the gradient of gradent
/*******************************************************************/
void surface_Reconstruction_Poisson_blending(Mat Z_approx, Mat X_gradient, Mat Y_gradient, Mat check_derivatives)
{
	Mat Laplacian_Z = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Laplacian_(Laplacian_Z, X_gradient, Y_gradient, check_derivatives);

	imshow("Laplacian_", Laplacian_Z);

	int vertices = Image.rows * Image.cols;

	Mat B = Mat(vertices, 1, CV_32F, Scalar(0));
	for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
		for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
			B.at<float>(rowIndex * Image.cols + colIndex, 0) = Laplacian_Z.at<float>(rowIndex, colIndex);
		}
	}

	Mat U = Mat(vertices, 1, CV_32F, Scalar(0));
	for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
		for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
			U.at<float>(rowIndex * Image.cols + colIndex, 0) = Z_approx.at<float>(rowIndex, colIndex);
		}
	}

	float initial_error = jacobi_Error(U, B);
	float new_error = initial_error;
	float loss = 1.0;

	/* Use Jacobi method to avoid superhigh dimensions and approximate Z */
	while (loss > 0.00005) {
		for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
			for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
				float RU = 0.0;
				if (rowIndex == 0 && colIndex == 0) {
					RU = U.at<float>(rowIndex*Image.cols + (colIndex + 1), 0) + U.at<float>((rowIndex + 1)*Image.cols + colIndex, 0);
				}
				else if (rowIndex == Image.rows - 1 && colIndex == Image.cols - 1) {
					RU = U.at<float>(rowIndex*Image.cols + (colIndex - 1), 0) + U.at<float>((rowIndex - 1)*Image.cols + colIndex, 0);
				}
				else if (rowIndex == 0 && colIndex != 0) {
					RU = U.at<float>(rowIndex*Image.cols + (colIndex + 1), 0) + U.at<float>((rowIndex + 1)*Image.cols + colIndex, 0) + U.at<float>(rowIndex*Image.cols + (colIndex - 1), 0);
				}
				else if (rowIndex == Image.rows - 1 && colIndex != Image.cols - 1) {
					RU = U.at<float>(rowIndex*Image.cols + (colIndex - 1), 0) + U.at<float>((rowIndex - 1)*Image.cols + colIndex, 0) + U.at<float>(rowIndex*Image.cols + (colIndex + 1), 0);
				}
				else {
					RU = U.at<float>(rowIndex*Image.cols + (colIndex + 1), 0) + U.at<float>((rowIndex + 1)*Image.cols + colIndex, 0) + U.at<float>(rowIndex*Image.cols + (colIndex - 1), 0) + U.at<float>((rowIndex - 1)*Image.cols + colIndex, 0);
				}
				U.at<float>(rowIndex * Image.cols + colIndex, 0) = -0.25 * (B.at<float>(rowIndex * Image.cols + colIndex, 0) - RU);
			}
		}
		new_error = jacobi_Error(U, B);
		loss = new_error / initial_error;
	}
	for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
		for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
			Z_approx.at<float>(rowIndex, colIndex) = U.at<float>(rowIndex * Image.cols + colIndex, 0);
		}
	}

	// make the surface bigger
	double min, max;
	cv::minMaxLoc(Z_approx, &min, &max);
	for (int rowIndex = 0; rowIndex < Image.rows; rowIndex++) {
		for (int colIndex = 0; colIndex < Image.cols; colIndex++) {
			Z_approx.at<float>(rowIndex, colIndex) = (Z_approx.at<float>(rowIndex, colIndex) - min) / (max - min) * 35;
		}
	}

	sharping(Z_approx, check_derivatives);
	medianBlur(Z_approx, Z_approx, 5);
}

int main()
{
	readImages();

	readLightSources();

	NormalImage = Mat(Image.rows, Image.cols, CV_32FC3);
	Mat X_gradient = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat Y_gradient = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Mat Z_approx = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));

	computeNormalandGradient(NormalImage, X_gradient, Y_gradient);

	Mat check_derivatives = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	sanity_check(check_derivatives, X_gradient, Y_gradient);

	surface_Reconstruction_Integration(Z_approx, X_gradient, Y_gradient, NormalImage);

	surface_Reconstruction_Poisson_blending(Z_approx, X_gradient, Y_gradient, check_derivatives);

	Mat Z = Mat(Image.rows, Image.cols, CV_32F, Scalar(0));
	Z_approx.copyTo(Z(Rect(0, 0, Image.cols, Image.rows)));

	writePly(Z, filename);

	imshow("CV", NormalImage);

	waitKey();

	return 0;
}



