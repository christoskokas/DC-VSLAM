cmake_minimum_required(VERSION 3.0.2)
set(CMAKE_CXX_STANDARD 17)
project(dc_vslam)

option(BUILD_WITH_ROS " Building with ROS " ON)


IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
# set(CMAKE_BUILD_TYPE Debug)
find_package(OpenCV 4.2.0)
if(NOT OpenCV_FOUND)
  find_package(OpenCV 3.2)
  if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 3.0 not found.")
  endif()
endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

## Find catkin and any catkin packages
if(BUILD_WITH_ROS)
find_package(catkin REQUIRED COMPONENTS 
  roscpp 
  rospy 
  std_msgs 
  genmsg 
  sensor_msgs
  pcl_ros
  pcl_conversions
  cv_bridge
  nav_msgs
  message_filters
  image_transport
)
find_package(PCL REQUIRED)

endif()
find_package(Pangolin REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Ceres REQUIRED)
find_package(yaml-cpp REQUIRED)

if(BUILD_WITH_ROS)
## Declare ROS messages and services
add_message_files(
  FILES
  AprilTagDetection.msg
  AprilTagDetectionArray.msg
)

add_library(${PROJECT_NAME} SHARED
src/Camera.cpp
src/Frame.cpp
src/Settings.cpp
src/FeatureExtractor.cpp
src/FeatureMatcher.cpp
src/KeyFrame.cpp
src/FeatureTracker.cpp
src/System.cpp
src/Map.cpp
src/LocalBA.cpp
include/Camera.h
include/Frame.h
include/Settings.h
include/FeatureExtractor.h
include/FeatureMatcher.h
include/KeyFrame.h
include/FeatureTracker.h
include/Map.h
include/System.h
include/Conversions.h
include/LocalBA.h
include/AprilTagDetection.h
include/AprilTagDetectionArray.h
)
else()
add_library(${PROJECT_NAME} SHARED
src/Camera.cpp
src/Frame.cpp
src/Settings.cpp
src/FeatureExtractor.cpp
src/FeatureMatcher.cpp
src/KeyFrame.cpp
src/FeatureTracker.cpp
src/System.cpp
src/Map.cpp
src/LocalBA.cpp
include/Camera.h
include/Frame.h
include/Settings.h
include/FeatureExtractor.h
include/FeatureMatcher.h
include/KeyFrame.h
include/FeatureTracker.h
include/Map.h
include/System.h
include/Conversions.h
include/LocalBA.h
)
endif()

if(BUILD_WITH_ROS)
## Generate added messages and services
generate_messages(DEPENDENCIES 
std_msgs
nav_msgs
)
catkin_package()
endif()

## Declare a catkin package
include_directories(
  ${catkin_INCLUDE_DIRS}  
  ${PCL_INCLUDE_DIRS} 
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${Pangolin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  )
if(BUILD_WITH_ROS)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
include_directories(
  ${PCL_INCLUDE_DIRS} 
  )
endif()
# add_executable(camera_subscriber src/camera_subscriber.cpp)
# target_link_libraries(camera_subscriber ${PCL_LIBRARIES} ${catkin_LIBRARIES})

# add_executable(Camera src/Camera.cpp)
# target_link_libraries(Camera ${PCL_LIBRARIES} ${catkin_LIBRARIES})

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${Pangolin_LIBRARIES}
${EIGEN3_LIBS}
${CERES_LIBRARIES}
${YAML_CPP_LIBRARIES}
)
if(BUILD_WITH_ROS)
target_link_libraries(${PROJECT_NAME}
${catkin_LIBRARIES}
${PCL_LIBRARIES}
)

add_executable(DualCamRT_gaz_exp src/DualCamRT_gaz_exp.cpp)
target_link_libraries(DualCamRT_gaz_exp ${PROJECT_NAME})

add_executable(DualCamRT_csl_exp src/DualCamRT_csl_exp.cpp)
target_link_libraries(DualCamRT_csl_exp ${PROJECT_NAME})

add_executable(DualCamRT_csl_exp_known src/DualCamRT_csl_exp_known.cpp)
target_link_libraries(DualCamRT_csl_exp_known ${PROJECT_NAME})

add_executable(DualCamRT src/DualCamRT.cpp)
target_link_libraries(DualCamRT ${PROJECT_NAME})

add_executable(SingleCamRT src/SingleCamRT.cpp)
target_link_libraries(SingleCamRT ${PROJECT_NAME})

add_executable(SingleCamRT_known src/SingleCamRT_known.cpp)
target_link_libraries(SingleCamRT_known ${PROJECT_NAME})

endif()

add_executable(SingleCam src/SingleCam.cpp)
target_link_libraries(SingleCam ${PROJECT_NAME})


add_executable(DualCam src/DualCam.cpp)
target_link_libraries(DualCam ${PROJECT_NAME})



