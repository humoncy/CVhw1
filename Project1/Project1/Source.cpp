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


const string Dir_path = "./test/";
const string Img_base = "Totoro";
const string Output_path = "./";

inline double abs_vector(Vec3d& tmp_v) {
	return sqrt((tmp_v.val[0])*(tmp_v.val[0])
		+ (tmp_v.val[1])*(tmp_v.val[1])
		+ (tmp_v.val[2])*(tmp_v.val[2]));
}

void photometric_stereo(const string Img_base, const unsigned int img_num, const int scale_height)
{
	cout << Img_base << endl;
	/************************read images & LightSource of images ******************************/
	string filename;
	Mat* pics_uchar = new Mat[img_num];
	for (unsigned int i = 1; i <= img_num; i++) {
		stringstream ss;
		ss << i;
		string NUM = ss.str();
		filename = Dir_path + Img_base + "/pic";
		filename += NUM;
		filename += ".bmp";
		pics_uchar[i - 1] = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
		if (pics_uchar[i - 1].empty()) {
			break;
		}
	}
	const int img_rows = pics_uchar[0].rows;
	const int img_cols = pics_uchar[0].cols;
	Mat val_S(img_num, 3, CV_64FC1);
	ifstream fptrText(Dir_path + Img_base + "/LightSource.txt");
	for (unsigned i = 0; i<img_num; ++i) {
		int tmp_int;
		fptrText.ignore(256, '(');
		fptrText >> tmp_int;
		val_S.at<double>(i, 0) = (double)tmp_int;
		fptrText.ignore(256, ',');
		fptrText >> tmp_int;
		val_S.at<double>(i, 1) = (double)tmp_int;
		fptrText.ignore(256, ',');
		fptrText >> tmp_int;
		val_S.at<double>(i, 2) = (double)tmp_int;
	}

	/***********************************normal surface with weight****************************************************************/
	Mat val_B(img_rows, img_cols, CV_64FC3);
	for (int j = 0; j<img_rows; j++) {
		for (int i = 0; i<img_cols; i++) {
			Mat val_I(img_num, 1, CV_64FC1);
			Mat weight(img_num, img_num, CV_64FC1, cvScalar(0.0));
			double mean = 0;
			for (unsigned int k = 0; k < img_num; k++) {
				val_I.at<double>(k, 0) = (double)pics_uchar[k].at<uchar>(j, i);
				if (val_I.at<double>(k, 0)>0 && val_I.at<double>(k, 0)<255)  weight.at<double>(k, k) = 1;
				else weight.at<double>(k, k) = 0.01;
			}
			Mat val_SW = weight*val_S;
			Mat val_SW_tr;
			transpose(val_SW, val_SW_tr);
			Mat val_trSWtr;
			invert(val_SW_tr*val_SW, val_trSWtr);
			val_trSWtr = val_trSWtr*val_SW_tr;
			Mat tmp_mat = val_trSWtr * weight *val_I;
			val_B.at<Vec3d>(j, i) = (Vec3d)tmp_mat;
		}
	}
	Mat val_N(img_rows, img_cols, CV_64FC3);
	//double abs_val_B_0 = abs_vector(val_B.at<Vec3d>(0, 0));
	//double min_abs_B = abs_val_B_0, max_abs_B = abs_val_B_0;
	Vec3d tmp_0(0.0, 0.0, 0.0);
	for (int j = 0; j<img_rows; j++) {
		for (int i = 0; i<img_cols; i++) {
			double abs_b = abs_vector(val_B.at<Vec3d>(j, i));
			//if (abs_b>max_abs_B)	 max_abs_B = abs_b;
			//else if (abs_b<min_abs_B)	min_abs_B = abs_b;
			if (abs_b == 0)  val_N.at<Vec3d>(j, i) = tmp_0;
			else val_N.at<Vec3d>(j, i) = (val_B.at<Vec3d>(j, i)) / abs_b;
		}
	}

	Mat CNfinal = val_N * 255;
	CNfinal.convertTo(CNfinal, CV_8UC3);
	imshow("Normal result", CNfinal);
	imwrite(Output_path + Img_base + ".bmp", CNfinal);
	waitKey(1);

	Mat Dx(img_rows, img_cols, CV_64F), Dy(img_rows, img_cols, CV_64F);
	for (int j = 0; j<img_rows; j++) {
		for (int i = 0; i<img_cols; i++) {
			Dx.at<double>(j, i) = -(val_N.at<Vec3d>(j, i)[0]) / (val_N.at<Vec3d>(j, i)[2] + 1);
			Dy.at<double>(j, i) = (val_N.at<Vec3d>(j, i)[1]) / (val_N.at<Vec3d>(j, i)[2] + 1);
		}
	}

	/******************************************integral*************************************************/

	Mat depth_map_V(img_rows, img_cols, CV_64F, cvScalar(0.0));
	Mat depth_map_H(img_rows, img_cols, CV_64F, cvScalar(0.0));
	int halfRow = img_rows / 2, halfCol = img_cols / 2;

	// construct depth_map_V
	depth_map_V.at<double>(halfRow, halfCol) = 0.0;
	for (int j = halfRow - 1; j >= 0; j--) 		depth_map_V.at<double>(j, halfCol) = depth_map_V.at<double>(j + 1, halfCol) - Dy.at<double>(j + 1, halfCol);
	for (int j = halfRow + 1; j<img_rows; j++)	depth_map_V.at<double>(j, halfCol) = depth_map_V.at<double>(j - 1, halfCol) + Dy.at<double>(j, halfCol);
	for (int j = 0; j<img_rows; j++) {
		for (int i = halfCol - 1; i >= 0; i--) 				depth_map_V.at<double>(j, i) = depth_map_V.at<double>(j, i + 1) - Dx.at<double>(j, i + 1);
		for (int i = halfCol + 1; i<img_cols; i++)	depth_map_V.at<double>(j, i) = depth_map_V.at<double>(j, i - 1) + Dx.at<double>(j, i);
	}
	// construct dep_map_H
	depth_map_H.at<double>(halfRow, halfCol) = 0.0;
	for (int i = halfCol - 1; i >= 0; i--)		depth_map_H.at<double>(halfRow, i) = depth_map_H.at<double>(halfRow, i + 1) - Dx.at<double>(halfRow, i + 1);
	for (int i = halfCol + 1; i<img_cols; i++)	depth_map_H.at<double>(halfRow, i) = depth_map_H.at<double>(halfRow, i - 1) + Dx.at<double>(halfRow, i);
	for (int i = 0; i<img_cols; i++) {
		for (int j = halfRow - 1; j >= 0; j--) 		depth_map_H.at<double>(j, i) = depth_map_H.at<double>(j + 1, i) - Dy.at<double>(j + 1, i);
		for (int j = halfRow + 1; j<img_rows; j++)	depth_map_H.at<double>(j, i) = depth_map_H.at<double>(j - 1, i) + Dy.at<double>(j, i);
	}
	double* depth_double = new double[img_rows*img_cols];
	for (int j = 0; j<img_rows; j++) {
		for (int i = 0; i<img_cols; i++) {
			depth_double[j*img_cols + i] = (depth_map_V.at<double>(j, i) + depth_map_H.at<double>(j, i)) / 2;
		}
	}
	/**********************************optimize************************************************/

	double* laplacian = new double[img_rows*img_cols];
	for (int j = 0; j<img_rows; j++) {
		for (int i = 0; i<img_cols; i++) {
			double Lx, Ly;
			if (i == 0) Lx = 0;
			else Lx = Dx.at<double>(j, i - 1) - Dx.at<double>(j, i);
			if (j == 0) Ly = 0;
			else Ly = Dy.at<double>(j - 1, i) - Dy.at<double>(j, i);

			laplacian[j*img_cols + i] = Lx + Ly;
		}
	}
	double* tmp_space = new double[img_rows*img_cols];
	double first_error = 1.0;
	double error = 1.0;
	for (unsigned int times = 0; (error / first_error)> 0.01; times++) {
		for (int j = 0; j<img_rows; j++) {
			for (int i = 0; i<img_cols; i++) {
				tmp_space[j*img_cols + i] = ldexp(depth_double[j*img_cols + i], 2);
				if (i != 0) {
					tmp_space[j*img_cols + i] = tmp_space[j*img_cols + i] - depth_double[j*img_cols + i - 1];
				}
				if (j != 0) {
					tmp_space[j*img_cols + i] = tmp_space[j*img_cols + i] - depth_double[(j - 1)*img_cols + i];
				}
				if (i != img_cols - 1) {
					tmp_space[j*img_cols + i] = tmp_space[j*img_cols + i] - depth_double[j*img_cols + i + 1];
				}
				if (j != img_rows - 1) {
					tmp_space[j*img_cols + i] = tmp_space[j*img_cols + i] - depth_double[(j + 1)*img_cols + i];
				}
			}
		}
		error = 0;
		for (int j = 0; j<img_rows; j++) {
			for (int i = 0; i<img_cols; i++) {
				depth_double[j*img_cols + i] = depth_double[j*img_cols + i] + ldexp(laplacian[j*img_cols + i] - tmp_space[j*img_cols + i], -2);
				error += fabs(tmp_space[j*img_cols + i] - laplacian[j*img_cols + i]);
			}
		}
		if (times == 0)	 first_error = error;
	}
	delete[] tmp_space;
	delete[] laplacian;

	/**************************************Normalize****************************************************/
	Mat depth_map(img_rows, img_cols, CV_64F, cvScalar(0.0));
	double min_depth = 0, max_depth = 0;
	for (int j = 0; j<img_rows; j++) {
		for (int i = 0; i<img_cols; i++) {
			depth_map.at<double>(j, i) = depth_double[j*img_cols + i];
			if (min_depth>depth_double[j*img_cols + i]) 		min_depth = depth_double[j*img_cols + i];
			else if (max_depth<depth_double[j*img_cols + i]) max_depth = depth_double[j*img_cols + i];
		}
	}
	delete[]depth_double;
	Mat depth_map_scale = (depth_map - min_depth) / (max_depth - min_depth)*scale_height;

	//filename = Output_path + Img_base + "_surface_HJ.ply";
	filename = "./surface_HJ.ply";
	ofstream foutPly(filename);
	foutPly << "ply\nformat ascii 1.0\ncomment alpha=1.0\nelement vertex " << img_rows*img_cols
		<< "\nproperty float x\nproperty float y\nproperty float z\n"
		<< "property uchar red\nproperty uchar green\nproperty uchar blue z\nend_header\n" << endl;
	for (int j = 0; j<img_rows; j++) {
		for (int i = 0; i<img_cols; i++) {
			foutPly << j << " " << i << " " << depth_map_scale.at<double>(j, i) << " 255 255 255\n";
		}
	}
	foutPly << endl;
	foutPly.close();

	delete[] pics_uchar;
	cv::destroyWindow("Normal result");
	return;
}
int main()
{
	photometric_stereo("star/", 6, 50);
}


