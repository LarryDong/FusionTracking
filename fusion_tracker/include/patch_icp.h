
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "patch.h"
#include "utilities.h"

namespace ICP{

using namespace std;
using namespace cv;

bool run_icp(vector<Point2f> vp, vector<Point2f> vq, Mat &R_inout, Mat &t_inout, int id = 1);

double find_match(const vector<Point2f> &vq, const vector<Point2f> &vp,
				  vector<Point2f> &vq_matched, vector<Point2f> &vp_matched, double outlier_thresh = 2);

void calc_R_t(const vector<Point2f> &vq, const vector<Point2f> &vp, Mat &R_qp, Mat &t_qp);

// // 计算ICP的误差，输入原始位置，和总的变换。
// float calculateICPError(const vector<Point2f> &vq, const vector<Point2f> &vp, 
// 	Mat &R_qp, Mat &t_qp,int outlier_thresh = 2);


}

