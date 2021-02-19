cmake_minimum_required(VERSION 2.8.3)
project(fusion_tracker)

find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  gflags_catkin
  dvs_msgs
  cv_bridge
  image_transport
  sensor_msgs
  std_msgs
  roscpp
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES fusion_tracker
#  CATKIN_DEPENDS roscpp std_msgs cv_bridge image_transport
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_node 
  src/fusion_tracker_node.cpp
  src/fusion_tracker.cpp
  src/fusion_tracker_utilities.cpp
  src/patch.cpp
  src/patch_icp.cpp
)

target_link_libraries(${PROJECT_NAME}_node  
  ${catkin_LIBRARIES}
  opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc opencv_video
  tbb
)

