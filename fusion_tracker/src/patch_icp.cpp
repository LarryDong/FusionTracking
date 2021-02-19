
#include "patch_icp.h"
#include <gflags/gflags.h>

// for calculate
#include <math.h>
#include <numeric>
#include <stdio.h>
#include <sys/time.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>
#include <mutex>
#include <thread>

DECLARE_double(icp_error_inliers);
DECLARE_double(icp_error_mse);
DECLARE_double(icp_inlier_change);
DECLARE_double(icp_mse_change);
DECLARE_double(icp_inlier_pixel);

int g_num = 5;


namespace ICP{


bool run_icp(vector<Point2f> vp, vector<Point2f> vq, Mat& R_inout, Mat& t_inout, int id){

	CV_Assert(R_inout.type() == CV_32FC1 && R_inout.size() == cv::Size(2, 2));
	CV_Assert(t_inout.type() == CV_32FC1 && t_inout.size() == cv::Size(1, 2));

	//
	Mat dR = Mat::eye(Size(2, 2), CV_32FC1);
	Mat dt = Mat::zeros(Size(1, 2), CV_32FC1);
	vector<Point2f> q_matched, p_matched;

	Mat R_total = Mat::eye(Size(2, 2), CV_32FC1);
	Mat t_total = Mat::zeros(Size(1, 2), CV_32FC1);

	double ave_error_new = -1, ave_error_old = -1;
	double inlier_ratio_old = -1, inlier_ratio_new = -1;
	int iter_counter = 0;
	bool ICP_converged = false;


	struct timeval tBegin, tEnd;
	double time_ms;


	for(int iter=0; iter< 20; ++iter){

		// ICP-0. Update errors.
		inlier_ratio_old = inlier_ratio_new;
		ave_error_old = ave_error_new;

		// ICP-1. Warp points by last R/t.
		tool::warpPoints(vp, dR, dt);


		// ICP-2. Find new matching.
		ave_error_new = find_match(vq, vp, q_matched, p_matched, FLAGS_icp_inlier_pixel);
		inlier_ratio_new = p_matched.size() * 1.0f / vp.size();
		

		// ICP-3. Calculate transform.
		calc_R_t(q_matched, p_matched, dR, dt);

		// ICP-4. Update R/t (from beginning to now)
		R_total = dR * R_total;
		t_total = dR * t_total + dt;

		iter_counter++;
		// ICP-5. Check convergence.
		if(abs(inlier_ratio_new - inlier_ratio_old)<FLAGS_icp_inlier_change
			&& abs(ave_error_new- ave_error_old)<FLAGS_icp_mse_change
			&& iter>=4){
			if(inlier_ratio_new > FLAGS_icp_error_inliers && ave_error_new < FLAGS_icp_error_mse){
				ICP_converged = true;
			}
			else{
				ICP_converged = false;
			}
			break;
		}
	}

	if(ICP_converged){
		R_inout = R_total.clone();
		t_inout = t_total.clone();
		return true;
	}
	else{
		return false;
	}
}


double find_match(const vector<Point2f> &vq, const vector<Point2f> &vp, 
	vector<Point2f> &vq_matched, vector<Point2f> &vp_matched, double outlier_thresh){

	vq_matched.clear();
	vp_matched.clear();
	vq_matched.reserve(vq.size());
	vp_matched.reserve(vp.size());

// #define USE_TBB
#ifndef USE_TBB
	// find every vp
	for(int i=0; i<vp.size(); ++i){
		Point2f p = vp[i];
		double minDist = 999;
		int minIndex = -1;
		for(int j=0; j<vq.size(); ++j){
			Point2f q = vq[j];
			double dist = (p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y);
			if(dist<minDist){
				minDist = dist;
				minIndex = j;
			}
		}
		// compare outliers.
		if (minDist < outlier_thresh * outlier_thresh){
			vp_matched.push_back(p);
			vq_matched.push_back(vq[minIndex]);
		}
	}
	double errort = 0;
	for(int i=0; i<vp_matched.size(); ++i){
		Point2f p = vp_matched[i];
		Point2f q = vq_matched[i];
		errort += sqrt((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y));
	}
	errort = errort / vq_matched.size();

#else
	// gettimeofday(&tBegin, NULL);
	// tbb::concurrent_vector<Point2f> con_vp_match, con_vq_match;
	tbb::concurrent_vector<Vec4f> con_vec4f;
	auto func = [&] (const tbb::blocked_range<int> &range){
		for(auto i=range.begin(); i!=range.end(); ++i){
			Point2f p = vp[i];
			double minDist = 999;
			int minIndex = -1;
			for(int j=0; j<vq.size(); ++j){
				Point2f q = vq[j];
				double dist = (p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y);
				if(dist < minDist){
					minDist = dist;
					minIndex = j;
				}
			}
			// compare outliers.
			if (minDist < outlier_thresh * outlier_thresh){		
				// con_vp_match.push_back(p);			// 先后push_back两个，顺序可能会出错。FUCK this!
				// con_vq_match.push_back(vq[minIndex]);
				Vec4f vec4f(p.x, p.y, vq[minIndex].x, vq[minIndex].y);
				con_vec4f.push_back(vec4f);
			}
		}
	};
	tbb::blocked_range<int> range(0, vp.size());
	tbb::parallel_for(range, func);

	int N = con_vec4f.size();
	vq_matched.resize(N);
	vp_matched.resize(N);
	for(int i=0; i<N; ++i){
		Vec4f v = con_vec4f[i];
		vp_matched[i] = Point2f(v[0], v[1]);
		vq_matched[i] = (Point2f(v[2], v[3]));
	}

#endif

	// calculate error;
	double error = 0;
	for(int i=0; i<vp_matched.size(); ++i){
		Point2f p = vp_matched[i];
		Point2f q = vq_matched[i];
		error += sqrt((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y));
	}
	error = error / vq_matched.size();
	return error;
}



// calculate best R and t between models
// @Input: vq, vp. Matched model sets/ data sets.
// @Output: R, t. Best R and t.
void calc_R_t(const vector<Point2f> &vq, const vector<Point2f> &vp, Mat &R_qp, Mat &t_qp){

	CV_Assert(R_qp.type()==CV_32FC1 && R_qp.size() == Size(2,2));
	CV_Assert(t_qp.type()==CV_32FC1 && t_qp.size() == Size(1,2));
	CV_Assert(vq.size() == vp.size());

	int N = vq.size();
	// Step 1. Calcualte mean of vq/vp
	Point2f vq_sum = accumulate(vq.begin(), vq.end(), Point2f(0,0));
	Point2f vp_sum = accumulate(vp.begin(), vp.end(), Point2f(0,0));

	Point2f vq_ave = Point2f(vq_sum.x / N, vq_sum.y / N);
	Point2f vp_ave = Point2f(vp_sum.x / N, vp_sum.y / N);

	vector<Point2f> vq_norm, vp_norm;
	vq_norm.resize(N);
	vp_norm.resize(N);

	// Step 2. Elimate mean. To norm.
	for(int i=0; i<N; ++i){
		vq_norm[i] = vq[i] - vq_ave;
		vp_norm[i] = vp[i] - vp_ave;
	}

	// Step 3. Calculate W = \Sigma q_i' * p_i
	Mat W = Mat::zeros(Size(2,2), CV_32FC1);
	Mat dW = Mat::zeros(Size(2,2), CV_32FC1);
	// (qx, qy) * (px, py)^T
	for(int i=0; i<N; ++i){
		double px, py, qx, qy;
		px = vp_norm[i].x;
		py = vp_norm[i].y;
		qx = vq_norm[i].x;
		qy = vq_norm[i].y;
		dW.at<float>(0, 0) = qx * px;
		dW.at<float>(0, 1) = qx * py;
		dW.at<float>(1, 0) = qy * px;
		dW.at<float>(1, 1) = qy * py;
		W = W + dW;
	}

	// Step 4. SVD, W = UDVt
	Mat U, Vt, D;
	SVD::compute(W, D, U, Vt, SVD::FULL_UV);
	CV_Assert(U.type() == CV_32FC1 && Vt.type() == CV_32FC1);
	
	// Step 5. R = UVt, t = mu_q - R * mu_p
	R_qp = U * Vt;

	double r11, r12, r21, r22;
	tool::mat2double(R_qp, r11, r12, r21, r22);

	t_qp.at<float>(0, 0) = vq_ave.x - (r11 * vp_ave.x + r12 * vp_ave.y);
	t_qp.at<float>(1, 0) = vq_ave.y - (r21 * vp_ave.x + r22 * vp_ave.y);
}

}

