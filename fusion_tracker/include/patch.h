
#pragma once

// ros
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

// utilities
#include <opencv2/opencv.hpp>

// events
#include "dvs_msgs/Event.h"
#include "dvs_msgs/EventArray.h"

// std
#include <iostream>
#include <deque>
#include <vector>

// program
#include "utilities.h"
#include <gflags/gflags.h>


enum Feature_Status{
	Tracking,
	Lost_OOB,
	Lost_Outliers,
	Lost_MatchFailed
};
class ModelPatch;
class DataPatch;

class ModelPatch{

public:
	ModelPatch(){ROS_FATAL("Empty init not allowed.");}
	ModelPatch(const cv::Mat& cannyRoi, cv::Point2f center, const cv::Mat& grayRoi);
	~ModelPatch(){;}
	ModelPatch& operator=(const ModelPatch& m);
	void calcSquare(void);			// calculate tl,tr,bl,br

	// for debug.
	cv::Mat getCannyImage();

	static int next_id_;
	int id_;
	cv::Point2f center_;				// Feature defined by it center and angle;
	double angle_;
	cv::Point2f tl_, tr_, bl_, br_;		// rectangle. Used when update.

	int batch_size_;					// number of edge pixels.
	std::vector<cv::Point2f> points_;	// points in global image.
	bool is_lost_;
	cv::Mat init_image_;
};



class DataPatch{
public:
	DataPatch() { ROS_FATAL("Not allow empty init."); }
	DataPatch(cv::Point2f center, int batch_size, int id, double time = 0.0f);
	DataPatch& operator=(const DataPatch& dp);

	bool accumulateEvent(dvs_msgs::Event e);
	bool updateOneEvent(const dvs_msgs::Event& e);	
	void updateAfterICP(const cv::Mat& R_last_curr, const cv::Mat& t_last_curr);
	
	void calcSquare(void);
	cv::Mat getCannyImage();

	int id_;
	int update_counter;		// counter to control update rate.
	int batch_size_;		// number of edge pixels.

	cv::Point2f center_, last_frame_center_;
	double angle_, last_frame_angle_;
	cv::Point2f tl_, bl_, tr_, br_;
	
	std::deque<cv::Point2f> events_position_;	// events position in global image.
	cv::Mat R_model_data_, t_model_data_;		// used for alignment (in ICP). From data 2 model R/t. in ICP

	// for record.
	double ts_;
	std::vector<double> time_history_;
	std::vector<cv::Point2f> center_history_;
	std::vector<double> angle_history_;
	Feature_Status status_;

	bool is_accumulated_;
	bool is_lost_;
	bool is_drawing_;
	int lifetime_after_lost_;
};


bool patchICP(const ModelPatch& mp, const DataPatch& dp, cv::Mat& R, cv::Mat& t);
