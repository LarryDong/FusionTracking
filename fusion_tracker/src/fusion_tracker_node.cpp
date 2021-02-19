
#include "fusion_tracker.h"
#include <gflags/gflags.h>

using namespace std;
using namespace cv;

// feature extraction parameters
DEFINE_int32(corner_number, 10, "Number of corner to track.");
DEFINE_int32(min_corners, 10, "Minimum features allowed to be tracked.");
DEFINE_int32(min_distance, 30, "Minimum distance between detected features. Parameter passed to goodFeatureToTrack.");
DEFINE_int32(block_size, 25, "Block size to compute Harris score. passed to harrisCorner and goodFeaturesToTrack.");
DEFINE_double(k, 0.04, "Magic number for Harris score.");
DEFINE_double(quality_level, 0.3, "Determines range of harris score allowed between the maximum and minimum. Passed to goodFeaturesToTrack.");
DEFINE_int32(canny_th1, 12, "Canny thresh");
DEFINE_int32(canny_th2, 24, "Canny thresh2");

// ICP parameters.
DEFINE_double(icp_events_ratio, 0.4, "Number of events(ratio) for calc ICP.");
DEFINE_double(icp_inlier_pixel, 2.5, "ICP match inlier distance.");
DEFINE_double(icp_error_inliers, 0.9, "Check ICP correct. If pattern the same, inliers should be larger than this ratio.");
DEFINE_double(icp_error_mse, 0.8, "Check ICP correct. Average error should be smaller than this ratio.");
DEFINE_double(icp_inlier_change, 0.05, "If converged, the change of inliers should be smaller 5%");
DEFINE_double(icp_mse_change, 0.1, "If converged, the change of mse should be smaller x pixels");

// image update parameters
DEFINE_double(image_update_velocity_th, 20, "Translate too fast. Larger than this value, no update.");
DEFINE_double(image_update_omega_th, 20, "Rotate too fast. Larger than this value, no update.");

// running control
DEFINE_int32(show_image, 1, "Show image when tracking.");
DEFINE_int32(update_features, 1, "Update features on new frames.");
DEFINE_int32(add_new_features, 1, "Add new features on new frames.");


int main(int argc, char** argv){

	google::ParseCommandLineFlags(&argc, &argv, true);
	ros::init(argc, argv, "fusion tracker node");
	ROS_INFO("tracker begin...");

	ros::NodeHandle nh;
	Tracker tracker(nh);
	ros::spin();

	return 0;
}

