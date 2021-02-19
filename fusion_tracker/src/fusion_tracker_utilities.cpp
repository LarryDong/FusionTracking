
// Some tracker functions are in this file.
// To make the process more clean.


#include "fusion_tracker.h"

using namespace std;
using namespace cv;

// imageCallback. only add images, not responsible for delete
void Tracker::imageCallback(const sensor_msgs::Image::ConstPtr &msg){
	// ROS_INFO("In image call back...");
    cv_bridge::CvImagePtr cv_ptr;
    try{
        cv_ptr = cv_bridge::toCvCopy(msg);
    }
    catch (cv_bridge::Exception& e){
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    std::unique_lock<std::mutex> lock(image_mutex_);

	Mat img = cv_ptr->image.clone();
	if (img.type() != CV_8UC1)
		cvtColor(img, img, COLOR_BGR2GRAY);
	image_data_buffer_.push_back(std::make_pair(msg->header.stamp, img));
	lock.unlock();

	// ROS_INFO_STREAM("Image buffer size: " << image_data_buffer_.size());
}


// 目前为止eventCallback只负责提供系统时间。
// 而且event array中第一个的时间戳，可能比当前image的时间戳要晚，
// 最后一个的时间戳，可能会比下一个的image时间戳还早（推测是下一个image传输较慢？）
void Tracker::eventCallback(const dvs_msgs::EventArray::ConstPtr& msg){
	std::unique_lock<std::mutex> lock(event_mutex_);
	for (const dvs_msgs::Event &e : msg->events){
		event_buffer_.push_back(e);
	}
	lock.unlock();

	// ROS_INFO_STREAM("eventCallback. Event buffer size: " << event_buffer_.size());
}



void Tracker::waitAndGetFirstImage(){
    ros::Rate r(10);

	// skip the first image to avoid imcomplete events data. (暂时跳过前两张图避免事件和图片的初始时间问题)
	bool avoid_the_first = true;		

    while (!got_first_image_){
        r.sleep();
        ROS_INFO("Waiting for first image.");
		std::unique_lock<std::mutex> images_lock(image_mutex_);

		if(avoid_the_first){
			if (image_data_buffer_.size() < 2) 	// 避免第一张图对应的事件时间戳不完整
				continue;
			image_data_buffer_.pop_front();

			image_data_.first = image_data_buffer_.front().first;
			image_data_.second = image_data_buffer_.front().second.clone();
			image_data_buffer_.pop_front();
			ROS_INFO_STREAM("Second image begin at: " << image_data_.first);
			got_first_image_ = true;
		}
		else{
			if (image_data_buffer_.empty())
				continue;
			
			image_data_.first = image_data_buffer_.front().first;
			image_data_.second = image_data_buffer_.front().second.clone();
			image_data_buffer_.pop_front();		// Bugs maybe here.
			ROS_INFO_STREAM("First image begin at: " << image_data_.first);
			got_first_image_ = true;
		}
    }
}



// deleta all events before the time.
bool Tracker::deleteEventsBeforeTime(ros::Time t){
	int counter = 0;
	dvs_msgs::Event e;

	while(1){
		std::unique_lock<std::mutex> lock(event_mutex_);
		if(event_buffer_.size()>0)
			e = event_buffer_.front();
		else {		// delete all events, but the last event is still before 't'
			ROS_WARN_STREAM("Deleted all " << counter << " events, but not up to time.");
			ROS_INFO_STREAM("Newset event at time: " << e.ts << ", delete time: " << t);
			return false;
		}
		if(e.ts<t){
			counter++;
			event_buffer_.pop_front();
		}
		else{
			ROS_INFO_STREAM("Deleted " << counter << " events before time: " << t);
			ROS_INFO_STREAM("Next event at time: " << e.ts);
			return true;
		}
	}
}

bool Tracker::deteteImagesBeforeTime(ros::Time t){
	int counter = 0;
	FrameData frame;
	while(1){
		std::unique_lock<std::mutex> lock(image_mutex_);
		if (image_data_buffer_.size() > 0)
			frame = image_data_buffer_.front();
		else{
			ROS_WARN_STREAM("Deleted all " << counter << " frames, but not up to time.");
			return false;
		}

		if(frame.first<t){
			counter++;
			image_data_buffer_.pop_front();
		}
		else{
			ROS_INFO_STREAM("Deleted " << counter << " frame before time: " << t);
			ROS_INFO_STREAM("Next event at time: " << frame.first);
			return true;
		}
	}
}



bool Tracker::getNewestImageBeforeTime(ros::Time t){
	std::unique_lock<std::mutex> images_lock(image_mutex_);

	if (image_data_buffer_.size() == 0)			// no image before time.
		return false;

	FrameData cur_image_data = image_data_buffer_.front();
	if (cur_image_data.first > t)
		return false;

	image_data_buffer_.pop_front();
	if (image_data_buffer_.size() == 0){			// only 1 image before time.
		last_image_data_.swap(image_data_);			// update last_image_data.
		image_data_.first = cur_image_data.first;
		image_data_.second = cur_image_data.second.clone();
		return true;
	}
	else{											// more than 1 image before time. Find the newest
		while(image_data_buffer_.size()!=0){
			FrameData next_image_data = image_data_buffer_.front();
			if(next_image_data.first > t){			// next image is after t. So current is the newest.
				break;
			}
			else{
				cur_image_data.first = next_image_data.first;
				cur_image_data.second = next_image_data.second.clone();
				image_data_buffer_.pop_front();
			}
		}
		last_image_data_.swap(image_data_);				// update last_image_data.
		image_data_.first = cur_image_data.first;		// update current_image_data.
		image_data_.second = cur_image_data.second.clone();
		return true;
	}
}



void Tracker::drawDataPatchInImage(const Mat& src){

	Mat showImg = src.clone();
	if (showImg.type() == CV_8UC1)
		cvtColor(showImg, showImg, COLOR_GRAY2BGR);

	for(DataPatch& dp:data_patch_vector_){

		if (!dp.is_drawing_)
			continue; // don't draw dp when marked no-drawing.
		
		Point2f lu = dp.tl_;
		Point2f ld = dp.bl_;
		Point2f ru = dp.tr_;
		Point2f rd = dp.br_;
		
		Scalar c = Scalar(255, 255, 255);
		switch (dp.status_){
			case Tracking:
				c = Scalar(0, 255, 0);
				break;
			case Lost_OOB:
				c = Scalar(0, 0, 255);
				break;
			case Lost_Outliers:
				c = Scalar(0, 255, 255);
				break;
			case Lost_MatchFailed:
				c = Scalar(255, 0, 0);
			default:
				break;
		}

		// drawing control.
		if(dp.status_!=Tracking)
			dp.lifetime_after_lost_++;
		if (dp.lifetime_after_lost_ > 500)
			dp.is_drawing_ = false;

		line(showImg, lu, ru, c);
		line(showImg, ru, rd, c);
		line(showImg, rd, ld, c);
		line(showImg, ld, lu, c);
	}
	namedWindow("Features", WINDOW_FREERATIO);
	imshow("Features", showImg);
	waitKey(1);
}



void Tracker::drawFeatureHistory(const Mat& src){
	Mat img = src.clone();
	if (img.type() != CV_8UC3)
		cvtColor(img, img, COLOR_GRAY2BGR);

	cv::RNG rng(time(0));
	for (DataPatch dp : data_patch_vector_){
		// Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		Scalar color = Scalar(0, 255, 0);
		if(dp.is_lost_){
			color = Scalar(0, 0, 255);
			continue;		// draw or not draw the lost trace.
		}
			
		int length = dp.center_history_.size();
		int max_len = 50;
		int begin_index = length > max_len ? length - max_len : 0;
		for (int i=begin_index; i<dp.center_history_.size()-1; ++i){
			Point2f p1 = dp.center_history_[i], p2 = dp.center_history_[i + 1];
			line(img, p1, p2, color, 1);
		}

		// draw current position
		Point2f ep =dp.center_history_.back();
		double angle = dp.angle_history_.back();
		circle(img, ep, 3, color);
		// int w = 10;
		// int dx = w * cos(angle);
		// int dy = w * sin(angle);
		// line(img, ep, Point(ep.x + dx, ep.y + dy), Scalar(0, 0, 255));
	}
	namedWindow("trace", WINDOW_FREERATIO);
	imshow("trace", img);
	waitKey(1);
}

