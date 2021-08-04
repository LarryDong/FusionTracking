# FusionTracking
> Feature tracking with standard and event-based camera fusion.

This repository provides a novel event-based feature tracking algorithm. This algorithm fuse data from both a standard camera and an event camera. Features are extracted, and updated by frames, and tracked from events. 
<center>
<figure>
<img src="https://raw.githubusercontent.com/LarryDong/FusionTracking/main/Pictures/method.png" />
</figure>
</center>


## Citaion
Please cite this paper if you use this code in an academic publication.

Paper avliable: [https://dl.acm.org/doi/abs/10.1145/3459066.3459075](https://dl.acm.org/doi/abs/10.1145/3459066.3459075)



## Build and Run

### 1. Dependency

First you need to install `gflags`, from [here](https://github.com/gflags/gflags)

`OpenCV` is also required (OpenCV 3.2.0 is tested).

### 2. Build

```bash
mkdir -p ~/tracking_ws/src
cd ~/tracking_ws/src
git clone git@github.com:LarryDong/FusionTracking.git
cd ..
catkin build
```

### 3. Run

We provide two methods. The first one we implemented based on the "ICP method" by Beat Kueng. The code of this method (only feature tracking part) is not released by the author so far.

```bibtex
@INPROCEEDINGS{ICP_Method,
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
or our fusion tracking method:
```bash
roslaunch fusion_tracker uft.launch
```

Then run the dataset.rosbag in a new terminal. Bags are from the [Event Camera Dataset](http://rpg.ifi.uzh.ch/davis_data.html).
```bash
rosbag play shape_6dof.bag
```


### 4. Trouble shooting
Tested under Ubuntu 18.04, ROS melodic, and OpenCV 3.2.0.


## Useful link
[1] Event Camera Dataset: [http://rpg.ifi.uzh.ch/davis_data.html](http://rpg.ifi.uzh.ch/davis_data.html)

[2] Celex_MP ROS driver: [https://github.com/kehanXue/CeleX5-ROS](https://github.com/kehanXue/CeleX5-ROS)

[3] Event feature tracking evaluation codes: [https://github.com/uzh-rpg/rpg_feature_tracking_analysis](https://github.com/uzh-rpg/rpg_feature_tracking_analysis)
This amazing tool can draw beautiful curves. The following pictures show the "feature age" and "tracking error" of 4 different method: ours, ICP method, EKLT, and EM-ICP. Please check the paper for more details.

<center>
<figure>
<img src="https://raw.githubusercontent.com/LarryDong/FusionTracking/main/Pictures/result.png" />
</figure>
</center>


## Remark
**This readme will be updated when the paper is formally published.**

