#pragma once

#include <deque>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <dvs_msgs/Event.h>
#include <string>


namespace tool{

using namespace cv;
using namespace std;

inline int min4(float a, float b, float c, float d){
	return (int)(min(min(min(a, b), c), d));
}
inline int max4(float a, float b, float c, float d){
	return (int)(max(max(max(a, b), c), d));
}

inline void mat2double(const cv::Mat& R, double& r11, double& r12, double& r21, double& r22){
	CV_Assert(R.type() == CV_32FC1 && R.size() == cv::Size(2, 2));
	r11 = R.at<float>(0, 0);
	r12 = R.at<float>(0, 1);
	r21 = R.at<float>(1, 0);
	r22 = R.at<float>(1, 1);
}

inline void vector2point(const cv::Mat& t, cv::Point2f& p){
	CV_Assert(t.type() == CV_32FC1 && t.size() == cv::Size(1, 2));
	p.x = t.at<float>(0,0);
	p.y = t.at<float>(1,0);
}

inline void vector2point(const cv::Mat& tMat, double& tx, double& ty){
	CV_Assert(tMat.type() == CV_32FC1 && tMat.size() == cv::Size(1, 2));
	tx = tMat.at<float>(0,0);
	ty = tMat.at<float>(1,0);
}


inline void point2vector(const cv::Point2f& p, cv::Mat& t){
	t = Mat::zeros(Size(1,2), CV_32FC1);
	t.at<float>(0, 0) = p.x;
	t.at<float>(1, 0) = p.y;
}

inline void point2vector(double tx, double ty, cv::Mat& t){
	t = Mat::zeros(Size(1,2), CV_32FC1);
	t.at<float>(0, 0) = tx;
	t.at<float>(1, 0) = ty;
}



inline void warpPoint(cv::Point2f& u, const cv::Mat& R, const cv::Mat&t){
	CV_Assert(R.type() == CV_32FC1 && R.size() == cv::Size(2, 2));
	CV_Assert(t.type() == CV_32FC1 && t.size() == cv::Size(1, 2));
	double r11, r12, r21, r22, tx, ty;

	vector2point(t, tx, ty);
	mat2double(R, r11, r12, r21, r22);
	cv::Point2f v;
	v.x = r11 * u.x + r12 * u.y + tx;
	v.y = r21 * u.x + r22 * u.y + ty;
	u = v;
}
inline void warpPoint(Point& p, const cv::Mat &R, const cv::Mat &t){
	Point2f pf = Point2f(p.x, p.y);
	warpPoint(pf, R, t);
	p = Point(int(pf.x), int(pf.y));
}
inline void warpPoint(double& x, double& y, const cv::Mat& R, const cv::Mat&t){
	CV_Assert(R.type() == CV_32FC1 && R.size() == cv::Size(2, 2));
	CV_Assert(t.type() == CV_32FC1 && t.size() == cv::Size(1, 2));

	Point2f pf = Point2f(x, y);
	warpPoint(pf, R, t);
	x = pf.x;
	y = pf.y;
}

inline void warpPoints(vector<Point2f>& pts, const Mat& R, const Mat& t){
	CV_Assert(R.type() == CV_32FC1 && R.size() == cv::Size(2, 2));
	CV_Assert(t.type() == CV_32FC1 && t.size() == cv::Size(1, 2));

	for(auto &p:pts)
		warpPoint(p, R, t);
}
inline void warpPoints(deque<Point2f>& pts, const Mat& R, const Mat& t){
	CV_Assert(R.type() == CV_32FC1 && R.size() == cv::Size(2, 2));
	CV_Assert(t.type() == CV_32FC1 && t.size() == cv::Size(1, 2));

	for(auto &p:pts)
		warpPoint(p, R, t);
}


inline void drawRect(Mat& src, Point2f lu, Point2f ru, Point2f rd, Point2f ld, cv::Scalar c=Scalar(0,255,0)){
	if (src.type() == CV_8UC1)
		cvtColor(src, src, COLOR_GRAY2BGR);
	line(src, lu, ru, c);
	line(src, ru, rd, c);
	line(src, rd, ld, c);
	line(src, ld, lu, c);
}

inline void coutLine(string sth, const Mat& R, const Mat& t){
	CV_Assert(R.size() == Size(2, 2));
	CV_Assert(t.size() == Size(1, 2));
	double r11, r12, r21, r22, tx, ty;
	mat2double(R, r11, r12, r21, r22);
	vector2point(t, tx, ty);
	ROS_INFO_STREAM(sth << "R: [" << r11 << ", " << r12 << "; " << r21 << " ," << r22 << "]. t: (" << tx << ", " << ty << ").");
}


inline double degree2rad(double angle){
	return (angle * CV_PI / 180.0f);
}
inline double rad2degree(double angle){
	return (angle * 180.0f / CV_PI);
}

inline double rotMat2angle(const Mat& R){
	double r11, r12, r21, r22, tx, ty;
	mat2double(R, r11, r12, r21, r22);
	// [ cos, sin]
	// [-sin, cos]
	double angle1 = atan2(-r12, r11);
	double angle2 = atan2(r21,  r22);

	return (angle1 + angle2) / 2;
}

inline Mat angle2rotMat(double angle){
	Mat rotMat(Size(2, 2), CV_32FC1);
	rotMat.at<float>(0, 0) = cos(angle);
	rotMat.at<float>(0, 1) = -sin(angle);
	rotMat.at<float>(1, 0) = sin(angle);
	rotMat.at<float>(1, 1) = cos(angle);
	return rotMat;
}


inline double pointCross(Vec2f p1, Vec2f p2){
	return p1[0]*p2[1]-p1[1]*p2[0];
}
inline bool checkInRect(const Point2f& p, const Point2f& lu, const Point2f& ru,const Point2f& rd,const Point2f& ld){
	// A      B
	//    p
	// D      C
	Vec2f V_ab = ru - lu;
	Vec2f V_cd = ld - rd;
	Vec2f V_ap = p - lu;
	Vec2f V_cp = p - rd;
	Vec2f V_ad = ld - lu;
	Vec2f V_cb = ru - rd;

	// ROS_INFO_STREAM("Rect: " << lu << ", " << ru << ", " << rd << ", " << ld << ".");
	
	return ( pointCross(V_ab, V_ap) * pointCross(V_cd, V_cp) >= 0 
			&& (pointCross(V_ad, V_ap) * pointCross(V_cb, V_cp) >= 0));
}
inline bool checkInImage(const Point2f& p, Size imageSize, int border=12){
	int x = p.x;
	int y = p.y;
	return (x >= border && y >= border && x <= imageSize.width - border && y <= imageSize.height - border);
}

inline double calcDistant2(const Point2f& p1, const Point2f& p2){
	return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}
inline double calcDistant(const Point2f& p1, const Point2f& p2){
	return sqrt(calcDistant2(p1, p2));
}


inline void showImage(string name, const Mat &src, int type = 1){
	if(type==1)
		namedWindow(name, WINDOW_FREERATIO);
	else
		namedWindow(name, WINDOW_NORMAL);
	
	imshow(name, src);
	waitKey(1);
}

inline void affine2Rt(const Mat& affine, Mat& R, Mat& t){
	CV_Assert(affine.size() == Size(3, 2) && affine.type() == CV_32FC1);
	
	double angle = atan2(affine.at<float>(0, 1), affine.at<float>(0, 0));
	R = angle2rotMat(angle);
	t = (Mat_<float>(2, 1) << affine.at<float>(0, 2), affine.at<float>(1, 2));
}
inline void Rt2Affine(const Mat& R, const Mat& t, Mat& warp){
	CV_Assert(R.size() == Size(2, 2) && R.type() == CV_32FC1);
	CV_Assert(t.size() == Size(1, 2) && t.type() == CV_32FC1);
	warp = Mat::zeros(Size(3, 2), CV_32FC1);
	warp.at<float>(0, 0) = R.at<float>(0, 0);
	warp.at<float>(0, 1) = R.at<float>(0, 1);
	warp.at<float>(1, 0) = R.at<float>(1, 0);
	warp.at<float>(1, 1) = R.at<float>(1, 1);
	warp.at<float>(0, 2) = t.at<float>(0, 0);
	warp.at<float>(1, 2) = t.at<float>(1, 0);
}

}

