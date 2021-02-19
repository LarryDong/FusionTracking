
#include "fusion_tracker.h"
#include <thread>
#include <gflags/gflags.h>
#include <string>
#include <map>
#include <math.h>
#include <iomanip>

DECLARE_int32(corner_number);
DECLARE_int32(min_corners);
DECLARE_int32(min_distance);
DECLARE_int32(block_size);
DECLARE_double(k);
DECLARE_double(quality_level);
DECLARE_double(icp_events_ratio);
DECLARE_double(icp_inlier_pixel);
DECLARE_int32(canny_th1);
DECLARE_int32(canny_th2);

DECLARE_double(image_update_velocity_th);
DECLARE_double(image_update_omega_th);

DECLARE_int32(show_image);
DECLARE_int32(add_new_features);
DECLARE_int32(update_features);


using namespace cv;
using namespace std;




class TimeDebugger{
public:
	TimeDebugger(){
		counter = 0;
		total_t = 0;
	}
	struct timeval tb, te;
	void begin(){
		gettimeofday(&tb, NULL);
	}
	double getTime(void){
		gettimeofday(&te, NULL);
		double t= 1000 * (te.tv_sec - tb.tv_sec) + (te.tv_usec - tb.tv_usec) / 1000.0f;
		counter++;
		total_t += t;
		return t;
	}
	int counter;
	double total_t;
	double getAverageTime(void){
		gettimeofday(&te, NULL);
		double t= 1000 * (te.tv_sec - tb.tv_sec) + (te.tv_usec - tb.tv_usec) / 1000.0f;
		counter++;
		total_t += t;
		return total_t / counter;
	}
	double getAll(void){
		return total_t;
	}
	void showInfo(string name){
		ROS_INFO_STREAM(name << ": all: " << total_t << ", counter: " << counter << ", ave: " << getAverageTime());
	}
};

TimeDebugger gt1, gt2, gt3, gt4, gt5;

Tracker::Tracker(ros::NodeHandle& nh){
	feature_number_ = 0;
	nh_ = nh;
	// ROS_INFO("Init tracker with nh");
	image_transport::ImageTransport it(nh_);
	sub_ = it.subscribe("/image", 10, &Tracker::imageCallback, this);
	event_sub_ = nh_.subscribe("/events", 10, &Tracker::eventCallback, this);

	got_first_image_ = false;

	// begin and detach the thread
	std::thread trackerThread(&Tracker::run, this);
    trackerThread.detach();
}


void Tracker::run(){

	// Step 1. init model patch sets
	initModelSet();

	// Step 2. init data point set
	feature_number_ = initDataSet();
	ROS_INFO_STREAM("Tracking: " << feature_number_ << " features...");

	// Step 3. update 
	ROS_INFO("Begin to track...");
	dvs_msgs::Event e;
	Mat R_old_new(Size(2, 2), CV_32FC1), t_old_new(Size(1, 2), CV_32FC1); // Rotation matrix from data to model
	bool first_time_flag = true;
	deleteEventsBeforeTime(curr_time_);		// have done this before (when init). Should delete nothing.
	deteteImagesBeforeTime(curr_time_);		// should delete images. Accmulate events needs time.
	
	TimeDebugger time_total, time_imageProcess, time_icp, time_drawing;
	int frame_counter = 0;
	double timer_1ms = ros::Time::now().toSec();
	int sys_time_int = 0;
	double sys_time_double = 0.0f;
	while(ros::ok()){
		// time_total.begin();
		double ros_running_time = ros::Time::now().toSec();
		if (ros_running_time - timer_1ms > 1){
			timer_1ms = ros_running_time;
			// ROS_ERROR_STREAM("Frame rate: " << frame_counter);
			frame_counter = 0;
		}

		if (!getOneEvent(e))		// newly added avoid empty.
			continue;
		
		sys_time_double = e.ts.toSec() - first_image_data_.first.toSec();
		if(sys_time_double > sys_time_int){
			sys_time_int++;
			ROS_WARN_STREAM("System running time: " << sys_time_double << " s");
		}

		if (getNewestImageBeforeTime(e.ts)){	// when new image comes
			frame_counter++;
			time_imageProcess.begin();
			// ROS_INFO_STREAM("New image at time: " << image_data_.first);
			if (first_time_flag)
				first_time_flag = false;
			else
				if(FLAGS_update_features)
					newImageProcess(); // process new image from the second.
				// time_imageProcess.showInfo("[Image]: ");
		}

		// process the event data
		for(int i=0; i<data_patch_vector_.size(); ++i){

			DataPatch& dp = data_patch_vector_[i];
			if (dp.is_lost_ )			// lost or update fail(not contain).
				continue;
			if(!dp.updateOneEvent(e))
				continue;

			if (!tool::checkInImage(dp.center_, image_data_.second.size(), FLAGS_block_size / 2)){
				// out of boundary. Stop tracking.
				dp.is_lost_ = true;
				dp.status_ = Lost_OOB;
				ROS_INFO_STREAM("Feature. " << dp.id_ << " lost [Out of image]. after: " << dp.time_history_.back() - dp.time_history_.front() << " s");
				feature_number_--;
				continue;
			}

			dp.update_counter++;		// control calculate fequency
			bool update_show_flag = false;		// control update show speed.
			static int show_control_slowdown = 0;
			if (dp.update_counter == (int)(dp.batch_size_ *FLAGS_icp_events_ratio)){ // update after a few events.
				dp.update_counter = 0;
				show_control_slowdown++;
				if(show_control_slowdown == 4){
					update_show_flag = true;
					show_control_slowdown = 0;
				}
				
				
				time_icp.begin();
				bool icp_succeed = patchICP(model_patch_vector_[i], dp, R_old_new, t_old_new);

				if(icp_succeed){
					dp.updateAfterICP(R_old_new, t_old_new);
				}
				else{
					dp.is_lost_ = true;
					dp.status_ = Lost_Outliers;
					ROS_INFO_STREAM("Feature. " << dp.id_ << " lost [ICP failed]. after: " << dp.time_history_.back() - dp.time_history_.front() << " s");
					feature_number_--;
				}
			}

			if(FLAGS_show_image){
				if (update_show_flag){		// drawing is real time-consuming!
					Mat cannyImg;
					Canny(image_data_.second, cannyImg, FLAGS_canny_th1, FLAGS_canny_th2, 3, true);
					cvtColor(cannyImg, cannyImg, COLOR_GRAY2BGR); 	// for drawing
					drawDataPatchInImage(cannyImg);
					drawFeatureHistory(image_data_.second);
				}
			}
			
		}
		// time_total.showInfo("[Loop once]: ");
	}

	while(1){;}
}



int Tracker::initModelSet(){
	ROS_INFO("Init model patch... ");

	// Step 1. Get the first image.
	waitAndGetFirstImage();
	curr_time_ = image_data_.first;			// update curr_time_ every time as possible.

	first_image_data_.first = image_data_.first;
	first_image_data_.second = image_data_.second.clone();
	ROS_INFO_STREAM("Use image at: " << first_image_data_.first << " as model.");

	// Step 2. Features extraction
	std::vector<cv::Point2d> corners;
	cv::goodFeaturesToTrack(first_image_data_.second, corners,
						FLAGS_corner_number,
						FLAGS_quality_level,
						FLAGS_min_distance, cv::Mat(),
						FLAGS_block_size,
						true,					// use Harris detector
						FLAGS_k);
	cv::Mat cannyImg;


	// Step 3. Model patch extraction
	cv::Canny(first_image_data_.second, cannyImg, FLAGS_canny_th1, FLAGS_canny_th2, 3, true);

	int width = image_data_.second.cols;
	int height = image_data_.second.rows;
	int hs = (FLAGS_block_size + 1) / 2;			// half size. to avoid boundary,
	model_patch_vector_.reserve(corners.size());
	for(Point2d pd:corners){
		Point p = Point(int(pd.x), int(pd.y));
		if (p.x <= hs || p.x >= width - hs || p.y <= hs || p.y >= height - hs)	// avoid boundary
			continue;
		Mat roi = cannyImg(Rect(p.x - hs, p.y - hs, FLAGS_block_size, FLAGS_block_size));
		Mat grayRoi = first_image_data_.second(Rect(p.x - hs, p.y - hs, FLAGS_block_size, FLAGS_block_size));
		ModelPatch patch(roi, p, grayRoi);			// construct patch
		model_patch_vector_.push_back(patch);		// saved to vector.
	}

	ROS_INFO_STREAM("Find " << corners.size() << " corners, and added " << model_patch_vector_.size() << " patches.");
	return model_patch_vector_.size();
}



int Tracker::initDataSet(){

	deleteEventsBeforeTime(first_image_data_.first);	// delete events before the first image.
	data_patch_vector_.reserve(model_patch_vector_.size());

	// construct DataPatch based on ModelPatch;
	int i = 0;
	ROS_INFO("Init data set.");
	for (ModelPatch &mp : model_patch_vector_){
		DataPatch dp(mp.center_, mp.batch_size_, mp.id_, curr_time_.toSec());
		data_patch_vector_.push_back(dp);
		ROS_INFO_STREAM("No. " << i++
							   << ", center: " << mp.center_
							   << ", batch size: " << dp.batch_size_);
	}

	// accumulate events.
	dvs_msgs::Event e;
	int accum_number = 0;
	while(true){		// extract events until all data patch are up to the batch size.
		accum_number = 0;
		if(getOneEvent(e)){		// extract one event.
			for(DataPatch& dp:data_patch_vector_){
				dp.accumulateEvent(e);		// accmulate for each data set.
				if(dp.is_accumulated_)
					accum_number++;
			}
			// check all the data patch set all full.
			if(accum_number == model_patch_vector_.size())
				break;
		};
	}
	curr_time_ = e.ts;
	ROS_INFO_STREAM("Data patch inited at: " << e.ts << ". Number: " << accum_number);
	return accum_number;
}



void Tracker::updateModelByLK(void){

	// STEP 1. Calculate Optic flow by LK. Use dp.center as init.
	vector<Point2f> last_pts, predict_pts, calc_pts;
	map<int, int> patch_index;

	gt1.begin();
	for (int i = 0; i < model_patch_vector_.size(); ++i){
		ModelPatch &mp = model_patch_vector_[i];
		DataPatch &dp = data_patch_vector_[i];

		last_pts.push_back(mp.center_); 		// don't care about lost_ now.
		predict_pts.push_back(dp.center_);

		if(dp.is_lost_){
			mp.is_lost_ = true; 				// set lost
			continue;
		}
	}
	if (last_pts.size() == 0){
		ROS_FATAL("Empty points size.");
		return ;
	}

	vector<float> errors;		// errors must be <float> type!
	vector<uchar> status;		// status must be <uchar> type!

	Mat lastImg = last_image_data_.second.clone();
	Mat currImg =  image_data_.second.clone();
	calc_pts = predict_pts;
	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	Size winSize(FLAGS_block_size, FLAGS_block_size);

	gt2.begin();
	calcOpticalFlowPyrLK(lastImg, currImg, last_pts,
						 calc_pts, status, errors,
						 winSize, 0, termcrit, OPTFLOW_USE_INITIAL_FLOW, 0.001);

	// STEP 2-3. Update ModelPatch and DataPatch.
	for (int i=0; i<model_patch_vector_.size(); ++i){
		ModelPatch &mp = model_patch_vector_[i];
		DataPatch &dp = data_patch_vector_[i];

		if (mp.is_lost_)
			continue;
	
		if (status[i] == 0)	// if not tracked by LK, not update. Use last position for next ICP.
			continue;
		
		// STEP 2. Update model patch.
		mp.center_ = calc_pts[i];
		mp.angle_ = dp.angle_;
		mp.calcSquare();

// #define USE_ECC				// Use enhanced correlation certiera to check image alignment. time-consuming
#ifdef USE_ECC
		// STEP 2.1. Check the same feature (Under homography assumption)
		bool match_failed = false;
		try{
			// int motion_assumption = MOTION_HOMOGRAPHY;		// may be parameters from input.
			double hs = FLAGS_block_size / 2;
			TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 0.5);
			Mat warp = (Mat_<float>(3, 3) << cos(mp.angle_), -sin(mp.angle_), mp.center_.x - hs,
							 sin(mp.angle_), cos(mp.angle_), mp.center_.y - hs,
							 0, 0, 1);
			findTransformECC(mp.init_image_, currImg, warp, MOTION_HOMOGRAPHY, criteria);
		}
		catch (cv::Exception){
			ROS_INFO("findTransformECC failed (MOTION_AFFINE). Feature lost.");
			ROS_INFO_STREAM("Feature. " << dp.id_ << " lost [ECC failed]. after: " << dp.time_history_.back() - dp.time_history_.front() << " s");
			dp.is_lost_ = true;
			dp.status_ = Lost_MatchFailed;
			mp.is_lost_ = true;
			continue;
		}
#endif

		// STEP 2.2. create mask to get new patch image.
		Mat cannyImg;
		Canny(image_data_.second, cannyImg, FLAGS_canny_th1, FLAGS_canny_th2, 3, true);
		Mat mask = Mat::zeros(cannyImg.size(), CV_8UC1);	// mask is used to extract region image.
		vector<Point> vpts;
		vpts.push_back(mp.tl_);
		vpts.push_back(mp.tr_);
		vpts.push_back(mp.br_);
		vpts.push_back(mp.bl_);
		vector<vector<Point>> vvpts;
		vvpts.push_back(vpts);
		fillPoly(mask, vvpts, 255);
		Mat canny_roi = cannyImg & mask;

		// STEP 2.3. Search in mask region. Avoid search the whole image.
		int col_min = tool::min4(mp.tl_.x, mp.tr_.x, mp.br_.x, mp.bl_.x);
		int col_max = tool::max4(mp.tl_.x, mp.tr_.x, mp.br_.x, mp.bl_.x);
		int row_min = tool::min4(mp.tl_.y, mp.tr_.y, mp.br_.y, mp.bl_.y);
		int row_max = tool::max4(mp.tl_.y, mp.tr_.y, mp.br_.y, mp.bl_.y);

		// STEP 2.4. Add points in mask.
		mp.points_.clear();		// clear, still reserve the memory.
		uchar *data = canny_roi.clone().data;		// .clone() is necessary.
		for (int i = row_min; i < row_max; ++i){
			for (int j = col_min; j < col_max; ++j){
				if (*(data + i * canny_roi.cols + j) > 128){
					Point p = Point(j, i);
					if (tool::checkInRect(p, mp.tl_, mp.tr_, mp.br_, mp.bl_)){
						mp.points_.push_back(p);
					}
				}
			}
		}
		
		// STEP 3. Update datapatch.
		// STEP 3.1. remove outliers. (Align)
		int outliers_cnt = 0;
		for(deque<Point2f>::iterator iter = dp.events_position_.begin(); iter!=dp.events_position_.end();){
			Point2f p = *iter;
			double minDist = 999;
			for (Point q : mp.points_){
				double dist = tool::calcDistant2(p, q);
				if (dist < minDist){
					minDist = dist;
				}
			}
			if (minDist > FLAGS_icp_inlier_pixel){		// check outliers
				iter = dp.events_position_.erase(iter);
				outliers_cnt++;
			}
			else
				iter++;
		}
		// ROS_INFO_STREAM("After Remove outliers: " << outliers_cnt << ", new batch size: " << dp.batch_size_ << ", events: " << dp.events_position_.size());

		// STEP 3.2. update translation between data and model.
		dp.R_model_data_ = Mat::eye(Size(2,2), CV_32FC1);					// R keeps unchanged (angle fixed.)
		dp.t_model_data_.at<float>(0) = mp.center_.x - dp.center_.x;		// t updates. (dp.center+t_model_data=mp.center)
		dp.t_model_data_.at<float>(1) = mp.center_.y - dp.center_.y;
	}
	return ;
}


int Tracker::addNewFeatures(void){
	int add_number = 0;
	int lost_number = 0, tracking_number = 0;
	Mat mask = Mat::zeros(first_image_data_.second.size(), CV_8UC1);
	int hs = FLAGS_block_size / 2;
	rectangle(mask, Rect(hs, hs, first_image_data_.second.cols - 2 * hs, first_image_data_.second.rows - 2 * hs), 255, -1); // avoid boundary.

	// get mask. and number.
	for (DataPatch dp : data_patch_vector_){
		if(!dp.is_lost_){
			tracking_number++;
			// set mask at each existing patch. using dp because mp is much delayed.
			// rectangle(mask, Rect(Point(dp.center_.x, dp.center_.y), Size(dp.patch_size_*2, dp.patch_size_*2)), 0, -1);
			circle(mask, dp.center_, FLAGS_block_size/2, 0, -1);
		}
	}

	if (tracking_number >= FLAGS_corner_number)
		return 0;

	std::vector<cv::Point2d> corners;
	goodFeaturesToTrack(image_data_.second, corners,
						FLAGS_corner_number - tracking_number,
						FLAGS_quality_level,
						FLAGS_min_distance,
						mask,
						FLAGS_block_size,
						true,
						FLAGS_k);
	
	if(corners.size() + feature_number_ <= FLAGS_min_corners){		// add features by if too less
		ROS_WARN("Too litte features. Try to add features by a lower threshold!");
		goodFeaturesToTrack(image_data_.second, corners,
			FLAGS_corner_number - tracking_number,
			FLAGS_quality_level/2,
			FLAGS_min_distance,
			mask,
			FLAGS_block_size,
			true,
			FLAGS_k/2);
	}
	if(corners.size() == 0){
		// ROS_INFO("No corners find.");
		return 0;
	}


	Mat cannyImg;
	Canny(image_data_.second, cannyImg, FLAGS_canny_th1, FLAGS_canny_th1, 3, true);

	for (Point2d pd : corners){
		Point p = Point(pd.x + 0.5, pd.y + 0.5);
		cv::Mat roi = cannyImg(cv::Rect(p.x - hs, p.y - hs, FLAGS_block_size, FLAGS_block_size));
		Mat grayRoi = image_data_.second(Rect(p.x - hs, p.y - hs, FLAGS_block_size, FLAGS_block_size));
		ModelPatch mp(roi, p, grayRoi);	
		DataPatch dp(p, mp.batch_size_, mp.id_, curr_time_.toSec());
		model_patch_vector_.push_back(mp);
		data_patch_vector_.push_back(dp);		// Don't need to accumulate now.
		add_number++;
	}

	ROS_INFO_STREAM("New feature added: " << add_number);
	return add_number;
}

void Tracker::calcFeatureSpeed(double &ave_v, double &ave_w){
	int feature_in_tracking = 0;
	double total_v = 0, total_w = 0;

	for (DataPatch &dp : data_patch_vector_){
		if(dp.is_lost_)
			continue;

		double delta_theta = dp.last_frame_angle_ - dp.angle_;
		Point2f pt = dp.last_frame_center_ - dp.center_;
		double delta_pixel = sqrt(pt.x * pt.x + pt.y * pt.y);

		total_v += delta_pixel;
		total_w += delta_theta;
		feature_in_tracking++;

		dp.last_frame_center_ = dp.center_;		// update last_frame
		dp.last_frame_angle_ = dp.angle_;
	}

	if(feature_in_tracking == 0){				// avoid divide-by-0.
		ROS_ERROR("No features are tracking...");		// should not happen.
		if(!FLAGS_add_new_features){
			ROS_INFO("Finished...");
			while(1);
		}
		ave_v = 999;		// assign a very big value.
		ave_w = 999;
		return;
	}
	ave_v = total_v / feature_in_tracking;
	ave_w = total_w / feature_in_tracking;
}


void Tracker::analyseFeatures(void){

	int n_track = 0, n_oob = 0, n_icp = 0, n_match = 0;
	for(DataPatch dp:data_patch_vector_){
		switch (dp.status_){
		case Tracking:
			n_track++;
			break;
		case Lost_OOB:
			n_oob++;
			break;
		case Lost_Outliers:
			n_icp++;
			break;
		case Lost_MatchFailed:
			n_match++;
		default:
			break;
		}
	}
	int n_total = data_patch_vector_.size();
	ROS_INFO_STREAM("Feature analysis: total(" << n_total << "), oob(" << n_oob << "), icp(" << n_icp << "), match(" << n_match << "). ");
}



void Tracker::newImageProcess(void){
	// update features if not moving fast.
	double ave_v, ave_w;
	calcFeatureSpeed(ave_v, ave_w); 		// calculate features' speed.
	double ave_degree = tool::rad2degree(ave_w);
	bool isMovingFast = false;
	if (ave_v <= FLAGS_image_update_velocity_th && abs(ave_degree) <= FLAGS_image_update_omega_th){
		updateModelByLK();
	}
	else{
		ROS_WARN_STREAM("Move too fast, no update features. v: " << ave_v << ", w: " << ave_degree << " (degree).");
		isMovingFast = true;
	}

	// add features if: not moving fast or too litte features;
	if(!isMovingFast || feature_number_<=FLAGS_min_corners){
	// if (feature_number_<=FLAGS_min_corners){
		if(FLAGS_add_new_features){
			int added_feature_number = addNewFeatures();
			feature_number_ += added_feature_number;
			// ROS_INFO_STREAM("Add new features: " << added_feature_number);
		}
	}
}


