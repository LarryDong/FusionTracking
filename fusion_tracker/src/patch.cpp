
#include "patch.h"
#include "patch_icp.h"

#include <gflags/gflags.h>
#include <math.h>
#include <numeric>

DECLARE_int32(block_size);

using namespace cv;
using namespace std;

int ModelPatch::next_id_ = 0;
ModelPatch::ModelPatch(const cv::Mat& img, cv::Point2f center, const Mat& grayRoi){

	if(img.type() != CV_8UC1)
		ROS_ERROR("Input type should be CV_8UC1");
	if (img.rows != FLAGS_block_size || img.cols != FLAGS_block_size){
		ROS_ERROR("Input patch size error!");
		ROS_ERROR_STREAM("Input size: (" << img.rows << ", " << img.cols << "), patch size: " << FLAGS_block_size);
	}

	batch_size_ = cv::countNonZero(img);
	center_ = center;
	angle_ = 0.0f;
	calcSquare();	// calcSquare by center and angle.

	points_.clear();
	points_.reserve(batch_size_);
	uchar *data = img.clone().data;		// 不加clone会出问题
	for(int i=0;i<img.rows;++i){
		for(int j=0; j<img.cols; ++j){
			if (*(data + i * img.cols + j) > 128)
				points_.push_back(cv::Point2f(j, i) + tl_);
		}
	}

	id_ = next_id_++;
	is_lost_ = false;
	init_image_ = grayRoi.clone();
}

ModelPatch& ModelPatch::operator=(const ModelPatch& m){
	if (this == &m)
		return *this;
	id_ = m.id_;
	center_ = m.center_;
	angle_ = m.angle_;
	batch_size_ = m.batch_size_;
	tl_ = m.tl_;
	tr_ = m.tr_;
	bl_ = m.bl_;
	br_ = m.br_;
	points_ = m.points_;
	init_image_ = m.init_image_.clone();
	return *this;
}

void ModelPatch::calcSquare(void){
	double hd = sqrt(2) / 2 * FLAGS_block_size;		// half diag.
	double theta = angle_ + CV_PI / 4;
	tl_ = center_ + hd * Point2f(-cos(theta), -sin(theta));
	bl_ = center_ + hd * Point2f(-sin(theta), cos(theta));
	tr_ = center_ + hd * Point2f(sin(theta), -cos(theta));
	br_ = center_ + hd * Point2f(cos(theta), sin(theta));
}

// for debug.
Mat ModelPatch::getCannyImage(){		// don't care about the speed.
	Mat cannyImg(Size(FLAGS_block_size, FLAGS_block_size), CV_8UC1);
	Mat wholeImg = Mat::zeros(Size(240, 180), CV_8UC1);

	// ROS_INFO_STREAM("NO. " << id_ << "Points size: " << points_.size());
	for(Point2f p:points_){	
		wholeImg.at<uchar>(p.y, p.x) = 255;
	}

// #define SHOW_PATCH
#ifdef SHOW_PATCH
	Mat patchImg = Mat::zeros(Size(patch_size_, patch_size_), CV_8UC1);
	vector<Point2f> vpts1, vpts2;
	vpts1.push_back(tl_);
	vpts1.push_back(bl_);
	vpts1.push_back(br_);
	vpts2.push_back(Point2f(0, 0));
	vpts2.push_back(Point2f(0, patch_size_));
	vpts2.push_back(Point2f(patch_size_, patch_size_));

	Mat rotMat = getAffineTransform(vpts1, vpts2);
	warpAffine(wholeImg, patchImg, rotMat, patchImg.size());
	return patchImg;
#else		// warp the image to local. May cause unclear edges because of interpolation.
	Scalar green = Scalar(0, 255, 0);
	cvtColor(wholeImg, wholeImg, COLOR_GRAY2BGR);
	circle(wholeImg, tl_, 3, Scalar(0,0,255), 1);
	circle(wholeImg, tr_, 3, green, 1);
	circle(wholeImg, bl_, 3, green, 1);
	circle(wholeImg, br_, 3, green, 1);
	line(wholeImg, tl_, bl_, green);
	line(wholeImg, bl_, br_, green);
	line(wholeImg, br_, tr_, green);
	line(wholeImg, tr_, tl_, green);
	return wholeImg;
#endif
}



//////////////////////////////    Data Patch     //////////////////////////////
//////////////////////////////    Data Patch     //////////////////////////////

DataPatch::DataPatch(cv::Point2f center, int batch_size, int id, double time){
	// new decided parameters/
	batch_size_ = batch_size;
	id_ = id;
	center_ = center;
	angle_ = 0.0f;
	calcSquare();
	last_frame_center_ =  center;
	last_frame_angle_ = 0.0f;
	
	R_model_data_ = Mat::eye(Size(2, 2), CV_32FC1);
	t_model_data_ = Mat::zeros(Size(1,2), CV_32FC1);

	update_counter = 0;
	is_lost_ = false;
	is_accumulated_ = false;

	// save history
	status_ = Tracking;
	ts_ = time;
	time_history_.push_back(time);
	center_history_.push_back(center);
	angle_history_.push_back(angle_);

	// for drawing.
	is_drawing_ = true;
	lifetime_after_lost_ = 0;

	// ROS_INFO_STREAM("DataPatch: " << id_ << " inited. Center: " << center_
	// 							  << ", Rect: " << tl_ << ", " << tr_ << ", " << br_ << ", " << bl_);
}


// 积累时，如果已经满了，则不需要积累；还是更新？这一个需要决定。暂时认为不需要更新。
bool DataPatch::accumulateEvent(dvs_msgs::Event e){
	if(is_accumulated_)		// no update.
		return false;
	
	Point2f p = Point2f(e.x, e.y);
	if(!tool::checkInRect(p, tl_, tr_, br_, bl_))		// not in DataPatch region.
		return false;

	ts_ = e.ts.toSec();				// save current time (Use `event.ts` as system time)
	events_position_.push_back(p);
	if(events_position_.size() == batch_size_){		// accmulate finished, output.
		ROS_INFO_STREAM("No." << id_ << " data_set has accumulated " << batch_size_);
		is_accumulated_ = true;
	}
	return true;
}


Mat DataPatch::getCannyImage(){

	Mat cannyImg(Size(FLAGS_block_size, FLAGS_block_size), CV_8UC1);
	Mat wholeImg = Mat::zeros(Size(240, 180), CV_8UC1);
	// ROS_INFO_STREAM("NO. " << id_ << "Points size: " << points_.size());

	for(Point2f p:events_position_)
		wholeImg.at<uchar>(int(p.y), int(p.x)) = 255;
	// ROS_INFO_STREAM("Unique pix: " << countNonZero(wholeImg));
	
// #define SHOW_PATCH
#ifdef SHOW_PATCH
	Mat patchImg = Mat::zeros(Size(patch_size_, patch_size_), CV_8UC1);
	vector<Point2f> vpts1, vpts2;
	vpts1.push_back(tl_);
	vpts1.push_back(bl_);
	vpts1.push_back(br_);
	vpts2.push_back(Point2f(0, 0));
	vpts2.push_back(Point2f(0, patch_size_));
	vpts2.push_back(Point2f(patch_size_, patch_size_));
	Mat rotMat = getAffineTransform(vpts1, vpts2);
	warpAffine(wholeImg, patchImg, rotMat, patchImg.size());
	return patchImg;
#else
	Scalar green = Scalar(0, 255, 0);
	cvtColor(wholeImg, wholeImg, COLOR_GRAY2BGR);
	circle(wholeImg, tl_, 3, Scalar(0,0,255), 1);
	circle(wholeImg, tr_, 3, green, 1);
	circle(wholeImg, bl_, 3, green, 1);
	circle(wholeImg, br_, 3, green, 1);
	line(wholeImg, tl_, bl_, green);
	line(wholeImg, bl_, br_, green);
	line(wholeImg, br_, tr_, green);
	line(wholeImg, tr_, tl_, green);
	return wholeImg;
#endif
}


DataPatch& DataPatch::operator=(const DataPatch& dp){
	id_ = dp.id_;
	batch_size_ = dp.batch_size_;

	// copy feature.
	center_ = dp.center_;
	angle_ = dp.angle_;
	last_frame_angle_ = dp.last_frame_angle_;
	last_frame_center_ =  dp.last_frame_center_;
	tl_ = dp.tl_;
	bl_ = dp.bl_;
	tr_ = dp.tr_;
	br_ = dp.br_;
	
	// copy flags
	is_accumulated_ = dp.is_accumulated_;
	is_lost_ = dp.is_lost_;
	is_drawing_ = dp.is_drawing_;
	lifetime_after_lost_ = dp.lifetime_after_lost_;

	// copy data.
	events_position_ = dp.events_position_;
	R_model_data_ = dp.R_model_data_.clone();
	t_model_data_ = dp.t_model_data_.clone();
	ts_ = dp.ts_;
	time_history_ = dp.time_history_;
	center_history_ = dp.center_history_;
	angle_history_ = dp.angle_history_;

	return *this;
}


bool DataPatch::updateOneEvent(const dvs_msgs::Event& e){

	Point2f p = Point2f(e.x, e.y);
	// ROS_INFO_STREAM("Update events: " << e.x << ", " << e.y);
	if(!tool::checkInRect(p, tl_, tr_, br_, bl_))
		return false;

	ts_ = e.ts.toSec();				// save current time

	// events_position_.size may changed when update dp. 
	// 1. outlier removed (after Model Patch updated.)
	// 2. new datapatch created.
	if(events_position_.size()==batch_size_){
		events_position_.push_back(p);
		events_position_.pop_front();
		return true;
	}
	
	if (events_position_.size() > batch_size_){
		events_position_.push_back(p);
		while (events_position_.size() > batch_size_)
			events_position_.pop_front();
		return true;
	}
	if (events_position_.size() < batch_size_){
		events_position_.push_back(p);
		return false;	// if size not large enough, may fail in ICP. So must reach the size.
	}
}


void DataPatch::calcSquare(void){		// use center+angle to update 4 rect.
	// double hd = sqrt(2) / 2 * patch_size_;		// half diag.
	double hd = sqrt(2) / 2 * FLAGS_block_size;
	double theta = angle_ + CV_PI / 4;
	tl_ = center_ + hd * Point2f(-cos(theta), -sin(theta));
	bl_ = center_ + hd * Point2f(-sin(theta), cos(theta));
	tr_ = center_ + hd * Point2f(sin(theta), -cos(theta));
	br_ = center_ + hd * Point2f(cos(theta), sin(theta));
}


bool patchICP(const ModelPatch& mp, const DataPatch& dp, cv::Mat& R_inout, cv::Mat& t_inout){

	CV_Assert(R_inout.type() == CV_32FC1 && R_inout.size() == cv::Size(2, 2));
	CV_Assert(t_inout.type() == CV_32FC1 && t_inout.size() == cv::Size(1, 2));

	// extract data.
	int N_model = mp.points_.size();
	int N_data = dp.events_position_.size();

	vector<Point2f> vp, vq;
	vp.resize(N_data);
	vq.resize(N_model);

	for (int i = 0; i < N_model; ++i){
		vq[i] = mp.points_[i];
	}
	for (int i = 0; i < N_data; ++i){
		Point2f p = dp.events_position_[i];
		tool::warpPoint(p, dp.R_model_data_, dp.t_model_data_);	// warp to model. Then icp.
		vp[i] = p;
	}

	return ICP::run_icp(vp, vq, R_inout, t_inout, mp.id_);
}


void DataPatch::updateAfterICP(const Mat& R_last_curr, const Mat& t_last_curr){
	// ROS_INFO("In update patch");
	CV_Assert(R_last_curr.type() == CV_32FC1 && t_last_curr.type() == CV_32FC1);
	CV_Assert(R_last_curr.size() == Size(2, 2) && t_last_curr.size() == Size(1, 2));
	
	// update center and angle.
	// new(u)->old(v): R*u+t = v
	// old->new: [R.t()]*v + [-R.t()*t] = u;
	Mat R = R_last_curr.clone();		// R/t, last->curr
	Mat t = -R.t() * t_last_curr;
	R = R.t();
	tool::warpPoint(center_, R, t);
	double delta_angle = tool::rotMat2angle(R);
	angle_ += delta_angle;

	// re calculate square based on center and angle.
	calcSquare();		// wartPoint for 4 vertices

	// Update data 2 model. Used for next ICP warp.
	// Rmd * (R*u + t) + tmd = (Rmd*R) * u + (Rmd*t + tmd);
	t_model_data_ = R_model_data_ * t_last_curr + t_model_data_;
	R_model_data_ = R_model_data_ * R_last_curr;

	// save history;
	time_history_.push_back(ts_);
	center_history_.push_back(center_);
	angle_history_.push_back(angle_);
}
