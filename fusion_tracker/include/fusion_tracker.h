
#pragma once

// std
#include <iostream>
#include <fstream>
#include <mutex>
#include <vector>
// ros
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
// opencv
#include <opencv2/opencv.hpp>
// events
#include "dvs_msgs/Event.h"
#include "dvs_msgs/EventArray.h"
// project
#include "patch.h"
#include "utilities.h"

// define buffer
using EventBuffer = std::deque<dvs_msgs::Event>;
using FrameData = std::pair<ros::Time, cv::Mat>;
using FrameBuffer = std::deque<FrameData>;


class Tracker{

public:
	Tracker(ros::NodeHandle& nh);

private:
	// main functions.
	void run();			// run the tracking thread.

	// init functions.
	int initModelSet();
	int initDataSet();
	// functions used for init.
	void waitAndGetFirstImage();
	bool getNewestImageBeforeTime(ros::Time t);
	bool deleteEventsBeforeTime(ros::Time t);
	bool deteteImagesBeforeTime(ros::Time t);
	void eventCallback(const dvs_msgs::EventArray::ConstPtr& msg);
	void imageCallback(const sensor_msgs::Image::ConstPtr &msg);

	// events update functions.
	inline bool getOneEvent(dvs_msgs::Event& e){
		std::unique_lock<std::mutex> lock(event_mutex_);
		if(event_buffer_.size()>0){
			e = event_buffer_.front();
			event_buffer_.pop_front();
			curr_time_ = e.ts;				// current time should be updated every time extract an event.
			return true;
		}
		else{
			// ROS_WARN("Event buffer empty.");		// this may happen after deleting the buffer.
			return false;
		}
	}

	// functions when new image comes.
	void newImageProcess(void);
	void updateModelByLK(void);			// update Data patch and model patch by KLT
	int addNewFeatures(void);			// add new features
	void calcFeatureSpeed(double &ave_v, double &ave_w);	// calculate all features' velocity/omega.
	void analyseFeatures(void);

	// for drawing & debugging.
	void drawDataPatchInImage(const cv::Mat& src);
	void drawFeatureHistory(const cv::Mat& src);


private:
	// ros
	ros::NodeHandle nh_;
	image_transport::Subscriber sub_;
	ros::Subscriber event_sub_;

	// data from rosbag
	FrameBuffer image_data_buffer_;		// image queue saved from rosbag
	EventBuffer event_buffer_;			// events queue saved from rosbag
	std::mutex image_mutex_;
	std::mutex event_mutex_;

	// data for tracking
	FrameData first_image_data_;
	bool got_first_image_;
	FrameData image_data_, last_image_data_;	// image always saved to image_data_
	ros::Time curr_time_;
	
	// features. 
	int feature_number_;
	std::vector<ModelPatch> model_patch_vector_;
	std::vector<DataPatch> data_patch_vector_;
	// std::ofstream feature_writer_;		// writer to save features. Deleted before release. 
};
