# DC-VSLAM
This is the software related to the publication: Multicamera Visual SLAM For Vineyard Inspection, submited at the CASE 2023.

DC-VSLAM is a multicamera visual SLAM designed for vineyard inspection. To address the challenge of homogeneous environments, loop closures are detected using AprilTags.

DC-VSLAM has been tested with OpenCV 4.2.0, Eigen 3.3.7 on Ubuntu 20.04 with ROS Noetic.

## Results

Kitti Dataset Sequence 01 :

![til](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExM2ZlM2YxMjRmYTNkMTQ0ODMxMmIwMjcwMjAyMWM3NjhlOGM0NzBkMSZjdD1n/xRn9qgUAySGaXSiVJT/giphy.gif)

EuRoC Dataset V1_02 :

![til](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYjgyYjQ2Mjk2YWUyMzNjMDJhYjIyYjM2NjU5MjUwMWQyZjMwMzA1MCZjdD1n/G59RVd1qs9gKQKNTmV/giphy.gif)


Dual Camera on Vineyard :

![til](https://media.giphy.com/media/bN2ff4O7f15VsyJL9p/giphy-downsized-large.gif)


# Installation

Two different ways are provided for installation. One is with ROS where DC-VSLAM can run real-time using rosbags, and another without ROS where DC-VSLAM runs using images. For the real-time operations ROS Noetic is needed.  Instructions on how to install ROS Noetic can be found here : http://wiki.ros.org/noetic/Installation.

## With ROS

Clone the package on the catkin workspace.

```
git clone https://github.com/christoskokas/DC-VSLAM.git
```

Download and install packages used by dc_vslam :

```
cd DC-VSLAM
chmod +x build.sh
./build.sh
```

From the catkin workspace install all the needed dependencies for the packages.

```
rosdep install --from-paths src --ignore-src -r -y
```

Build the workspace.

```
catkin_make
source devel/setup.bash
```

## Without ROS

Build both dc_vslam and its packages without ROS :

```
chmod +x build_no_ros.sh
./build_no_ros.sh
```

# Quick Start

Several launch files are provided. The RT denotes real-time and the AT denotes the use of AprilTag Loop Closure. Change the launch files to match the config file name and the topic of the image msgs for AprilTag detection.

DC-VSLAM can run both with images and with rosbags. Images need to be provided as presented below ( the bullets are folders ): 

- DC_VSLAM
  - images
    - dataset_name
      - left
        1. 000000.jpg(.png)
        2. 000001.jpg(.png)
        3. ...
      - right
        1. 000000.jpg(.png)
        2. 000001.jpg(.png)
        3. ...
      - leftBack
        1. 000000.jpg(.png)
        2. 000001.jpg(.png)
        3. ...
      - rightBack
        1. 000000.jpg(.png)
        2. 000001.jpg(.png)
        3. ...

And the full path to the dataset folder has to be provided in the config file.

### With ROS

To run Dual Cam with apriltag_ros package ( for Loop Closure ) at the same time :

```
roslaunch dc_vslam DualCamRTAT.launch
```

To run Dual Cam with AprilTag Detcetion messages ( for Loop Closure ) in a topic :

```
roslaunch dc_vslam DualCamRT.launch
```

To run Dual Cam from images in a directory :

```
roslaunch dc_vslam DualCam.launch
```

For Single Cam either from rosbag :

```
roslaunch dc_vslam SingleCamRT.launch
```

or images : 

```
roslaunch dc_vslam SingleCam.launch
```

### Without ROS

If the installation was completed without ROS images need to be provided. Single Cam is also provided to test on well-known datasets like KITTI or EUROC. Configs for both KITTI and EUROC are provided. To run single cam :

```
./SingleCam config_file_name
```

with the appropriate changes on the config file.

To run Dual cam : 

```
./DualCam config_file_name
```

## TO DO

- [] Optimize Code
- [] Add IMU integration
- [] Add more Apriltags

