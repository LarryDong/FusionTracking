# FusionTracking
> Feature tracking with standard and event-based camera fusion.

This repository provides a novel event-based feature tracking algorithm. This algorithm fuse data from both a standard camera and an event camera. Features are extracted, and updated by frames, and tracked from events. Please cite the paper if you use this code.

## Citaion
Paper is not formally published now. 



## Build and Run

1. Dependency
First you need to install `gflags`, from [here](https://github.com/gflags/gflags)
`Opencv` is also needed.

2. Build
```bash
mkdir -p ~/tracking_ws/src
cd ~/tracking_ws/src
git clone git@github.com:LarryDong/FusionTracking.git
cd ..
catkin build
```

3. Run
We provide two methods. The first one implements "ICP method" by Beat Kueng. Please cite this paper:

```bibtex
@INPROCEEDINGS{7758089,
  author={B. {Kueng} and E. {Mueggler} and G. {Gallego} and D. {Scaramuzza}},
  booktitle={2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Low-latency visual odometry using event-based feature tracks}, 
  year={2016},
  volume={},
  number={},
  pages={16-23},
  doi={10.1109/IROS.2016.7758089}}
```

We improve this method by adding frame information.

Before launch, you should source the `setup.bash` and modify the `--flagfile` path in each .launch file.

To run the "ICP method":
```bash
roslaunch fusion_tracker icp.launch
```
or this fusion tracking method:
```bash
roslaunch fusion_tracker uft.launch
```

Then run the rosbag. Bags are from the [Event Camera Dataset](http://rpg.ifi.uzh.ch/davis_data.html).
```bash
rosbag play shape_6dof.bag
```


4. Problems shooting
Tested under Ubuntu 18.04, Ros melodic, and OpenCV 3.2.0.


5. Useful link
Event Camera Dataset: http://rpg.ifi.uzh.ch/davis_data.html
Celex_MP ros driver: https://github.com/kehanXue/CeleX5-ROS/
Event feature tracking evaluation codes: https://github.com/uzh-rpg/rpg_feature_tracking_analysis

!! This readme will be updated when the paper is formally published. !!

