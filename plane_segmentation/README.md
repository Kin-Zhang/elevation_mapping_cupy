# Convex Terrain Representation #

## Overview
This is a C++ ROS package for extracting convex polygons from elevation maps created by elevation_mapping.  

## Installation

### Dependencies

#### OpenCV
Make sure you have openCV installed.  
You can execute the following command to install it.
```bash
sudo apt-get install libopencv-dev
```

#### Eigen
```bash
sudo apt install libeigen3-dev
```

#### CGAL
CGAL5 is required. It will be automatically downloaded and installed into the catkin workspace by the cgal5_catkin package.
Make sure you have the third-party libaries installed on you machine:
```bash
sudo apt-get install libgmp-dev
sudo apt-get install libmpfr-dev
sudo apt-get install libboost-all-dev
```

#### PCL
PCL is required, but the ANYbotics distributed version does not contain visualization components. 
With the following commands PCL can be build from source directly into your catkin workspace. 
DO NOT do this on the ANYmal onboard PCs, only on OPC and simulation PCs.
```bash
sudo apt-get install libvtk7-dev
catkin build pcl_catkin
```

### ROS package dependencies

#### JSK-visualization
For rviz-visualization the jsk-library is used.
```bash
sudo apt-get install ros-noetic-jsk-visualization
```

#### Grid Map
Grid map is available through ANYmal-research, or you can add it to your workspace. You can clone it using:
```bash
git clone https://github.com/ANYbotics/grid_map.git
```

## Usage
### Build
```bash
catkin build convex_plane_decomposition_ros
```
### Run demo
```bash
roslaunch convex_plane_decomposition_ros demo.launch
```

### Parameters
You can select input map topics, pipeline parameters etc. in the respective yaml file in
```bash
convex_plane_decomposition_ros/config/
```
Some other parameters are set through the launch files.