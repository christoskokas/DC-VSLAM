#include "Settings.h"
#include "System.h"
#include "Camera.h"
#include "Frame.h"
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <std_msgs/Int64.h>
#include <std_srvs/SetBool.h>
#include <std_msgs/String.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <iostream>
#include <sstream>
#include <string>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <boost/foreach.hpp>
#include <thread>
#include <yaml-cpp/yaml.h>
#include <signal.h>
#include <AprilTagDetection.h>
#include <AprilTagDetectionArray.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>

#define GTPOSE false
#define PUBPOINTCLOUD false
#define ATBESTPOSE false

class GetImagesROS
{
    public:
        GetImagesROS(DC_VSLAM::System* _voSLAM) : voSLAM(_voSLAM), ac("move_base", true){};

        void getImages(const sensor_msgs::ImageConstPtr& msgLeft,const sensor_msgs::ImageConstPtr& msgRight, const sensor_msgs::ImageConstPtr& msgLeftB,const sensor_msgs::ImageConstPtr& msgRightB,const nav_msgs::OdometryConstPtr& msgGT);
        void getImages(const sensor_msgs::ImageConstPtr& msgLeft,const sensor_msgs::ImageConstPtr& msgRight, const sensor_msgs::ImageConstPtr& msgLeftB,const sensor_msgs::ImageConstPtr& msgRightB);
        void aprilTagCallBack(const DC_VSLAM::AprilTagDetectionArray::ConstPtr& msg);
        void currGoalCallBack(const geometry_msgs::PoseStamped::ConstPtr& msg);

        void saveGTTrajectoryAndPositions(const std::string& filepath, const std::string& filepathPosition);
        void publishOdom(const std_msgs::Header& _header);
        void pubGoalAT(const std_msgs::Header& _header, const Eigen::Matrix4d& optPose);

        ros::Subscriber aprilTag_sub;
        ros::Subscriber currGoal_sub;

        ros::Publisher pc_pub;
        ros::Publisher odom_pub;
        tf::TransformBroadcaster odom_broadcaster;

        DC_VSLAM::System* voSLAM;

        std::vector<Eigen::Vector3d> gtPositions;
        std::vector<Eigen::Quaterniond> gtQuaternions;

        Eigen::Matrix4d T_w_c1;
        Eigen::Matrix4d T_c1_AT = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d tagPose;
        Eigen::Matrix4d tagPoseW;
        Eigen::Matrix4d T_tag_b;
        Eigen::Matrix4d T_bf_c1;
        Eigen::Matrix4d optPose;
        Eigen::Matrix4d prevGoal = Eigen::Matrix4d::Identity();

        actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> ac;


        cv::Mat rectMap[2][2];
        cv::Mat rectMapB[2][2];
        int frameNumb {0};
        int ATFoundCount {0};
        int ATLostCount {0};
        bool ATFound {false};
        bool tagPoseFilled {false};
        bool onTrack {false};
        bool first {true};
        bool pathSuccess {false};
        bool gotPrevGoal {false};
        bool poseOptAT {false};
        bool pathsucRep {true};
};

void signal_callback_handler(int signum) {
    std::cout << "Caught signal " << signum << std::endl;
    ros::shutdown();
    // Terminate program
}

int main (int argc, char **argv)
{
    ros::init(argc, argv, "Dual_Cam");
    ros::start();
    signal(SIGINT, signal_callback_handler);

    std::string file = "config_gaz_exp.yaml";
    DC_VSLAM::ConfigFile* confFile = new DC_VSLAM::ConfigFile(file.c_str());

    if ( confFile->badFile )
        return -1;

    DC_VSLAM::System* voSLAM = new DC_VSLAM::System(confFile, true, true);

    GetImagesROS imgROS(voSLAM);

    DC_VSLAM::Zed_Camera* mZedCamera = voSLAM->mZedCamera;
    DC_VSLAM::Zed_Camera* mZedCameraB = voSLAM->mZedCameraB;

    mZedCamera->numOfFrames = INT_MAX;
    mZedCameraB->numOfFrames = INT_MAX;


    const int width = mZedCamera->mWidth;
    const int height = mZedCamera->mHeight;

    if ( !mZedCamera->rectified )
    {
        cv::Mat R1,R2;
        cv::initUndistortRectifyMap(mZedCamera->cameraLeft.K, mZedCamera->cameraLeft.D, mZedCamera->cameraLeft.R, mZedCamera->cameraLeft.P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, imgROS.rectMap[0][0], imgROS.rectMap[0][1]);
        cv::initUndistortRectifyMap(mZedCamera->cameraRight.K, mZedCamera->cameraRight.D, mZedCamera->cameraRight.R, mZedCamera->cameraRight.P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, imgROS.rectMap[1][0], imgROS.rectMap[1][1]);

    }

    if ( !mZedCameraB->rectified )
    {
        cv::Mat R1,R2;
        cv::initUndistortRectifyMap(mZedCameraB->cameraLeft.K, mZedCameraB->cameraLeft.D, mZedCameraB->cameraLeft.R, mZedCameraB->cameraLeft.P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, imgROS.rectMapB[0][0], imgROS.rectMapB[0][1]);
        cv::initUndistortRectifyMap(mZedCameraB->cameraRight.K, mZedCameraB->cameraRight.D, mZedCameraB->cameraRight.R, mZedCameraB->cameraRight.P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, imgROS.rectMapB[1][0], imgROS.rectMapB[1][1]);

    }

    cv::Mat im(mZedCamera->mHeight, mZedCamera->mWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::namedWindow("VSLAM : Front Camera1");
    cv::imshow("VSLAM : Front Camera1", im);
    cv::waitKey(1);

    cv::namedWindow("VSLAM : Back Camera");
    cv::imshow("VSLAM : Back Camera", im);
    cv::waitKey(1000);


    bool gtPose {false};
    try
    {
        gtPose = confFile->getValue<bool>("gtPose");
    }
    catch(const std::exception& e)
    {
        gtPose = false;
    }
    bool LCSameCam {true};
    try
    {
        LCSameCam = confFile->getValue<bool>("LCSameCam");
    }
    catch(const std::exception& e)
    {
        LCSameCam = true;
    }
    if ( !LCSameCam )
    {
        std::vector<double> Tc1AT = confFile->getValue<std::vector<double>>("T_c1_AT","data");
        imgROS.T_c1_AT << Tc1AT[0],Tc1AT[1],Tc1AT[2],Tc1AT[3],Tc1AT[4],Tc1AT[5],Tc1AT[6],Tc1AT[7],Tc1AT[8],Tc1AT[9],Tc1AT[10],Tc1AT[11],Tc1AT[12],Tc1AT[13],Tc1AT[14],Tc1AT[15];
    }

    // This is for when you know where the tag is. Changes need to be made in LC to account for that

    // bool  tagPose {false};
    // try
    // {
    //     tagPose = confFile->getValue<bool>("tagPose");
    // }
    // catch(const std::exception& e)
    // {
    //     tagPose = false;
    // }
    // if ( tagPose )
    // {
    //     std::vector<double> Twtag = confFile->getValue<std::vector<double>>("T_w_tag","data");
    //     imgROS.tagPose << Twtag[0],Twtag[1],Twtag[2],Twtag[3],Twtag[4],Twtag[5],Twtag[6],Twtag[7],Twtag[8],Twtag[9],Twtag[10],Twtag[11],Twtag[12],Twtag[13],Twtag[14],Twtag[15];
    // }


    const std::string gtPath = (gtPose) ? confFile->getValue<std::string>("gtPath") : "";

    const std::string leftPath = confFile->getValue<std::string>("leftPathCam");
    const std::string rightPath = confFile->getValue<std::string>("rightPathCam");
    const std::string leftPathB = confFile->getValue<std::string>("leftPathCamB");
    const std::string rightPathB = confFile->getValue<std::string>("rightPathCamB");
    const std::string aprilTagPath = confFile->getValue<std::string>("aprilTagPath");

    ros::NodeHandle nh;
    
    message_filters::Subscriber<sensor_msgs::Image> left_sub(nh, leftPath, 100);
    message_filters::Subscriber<sensor_msgs::Image> right_sub(nh, rightPath, 100);
    message_filters::Subscriber<sensor_msgs::Image> left_subB(nh, leftPathB, 100);
    message_filters::Subscriber<sensor_msgs::Image> right_subB(nh, rightPathB, 100);
    imgROS.aprilTag_sub = nh.subscribe(aprilTagPath, 10, &GetImagesROS::aprilTagCallBack, &imgROS);

#if ATBESTPOSE

    std::vector<double> Tbtag = confFile->getValue<std::vector<double>>("T_tag_b","data");
    imgROS.T_tag_b << Tbtag[0],Tbtag[1],Tbtag[2],Tbtag[3],Tbtag[4],Tbtag[5],Tbtag[6],Tbtag[7],Tbtag[8],Tbtag[9],Tbtag[10],Tbtag[11],Tbtag[12],Tbtag[13],Tbtag[14],Tbtag[15];

    std::vector<double> Tbfc1 = confFile->getValue<std::vector<double>>("T_bf_c1","data");
    imgROS.T_bf_c1 << Tbfc1[0],Tbfc1[1],Tbfc1[2],Tbfc1[3],Tbfc1[4],Tbfc1[5],Tbfc1[6],Tbfc1[7],Tbfc1[8],Tbfc1[9],Tbfc1[10],Tbfc1[11],Tbfc1[12],Tbfc1[13],Tbfc1[14],Tbfc1[15];

    imgROS.currGoal_sub = nh.subscribe("/move_base/current_goal", 10, &GetImagesROS::currGoalCallBack, &imgROS);
#endif
    
#if PUBPOINTCLOUD
    imgROS.odom_pub = nh.advertise<nav_msgs::Odometry>("/odom",1);
    std::vector<double> Twc1 = confFile->getValue<std::vector<double>>("T_w_c1","data");
    imgROS.T_w_c1 << Twc1[0],Twc1[1],Twc1[2],Twc1[3],Twc1[4],Twc1[5],Twc1[6],Twc1[7],Twc1[8],Twc1[9],Twc1[10],Twc1[11],Twc1[12],Twc1[13],Twc1[14],Twc1[15];
    imgROS.pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/DC_VSLAM/points2",1);
#endif
#if GTPOSE
    message_filters::Subscriber<nav_msgs::Odometry> gt_sub(nh, gtPath, 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, nav_msgs::Odometry> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(100), left_sub,right_sub, left_subB,right_subB, gt_sub);
    sync.registerCallback(boost::bind(&GetImagesROS::getImages,&imgROS,_1,_2, _3,_4, _5));
#else
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), left_sub,right_sub, left_subB,right_subB);
    sync.registerCallback(boost::bind(&GetImagesROS::getImages,&imgROS,_1,_2, _3,_4));


#endif
    std::cout << "Waiting for images.. " << std::endl;
    ros::spin();

    std::cout << "System Shutdown!" << std::endl;
    voSLAM->exitSystem();
    std::cout << "Saving Trajectory.." << std::endl;
    voSLAM->saveTrajectoryAndPosition("dual_cam_traj.txt", "dual_cam_pos.txt");
#if GTPOSE
    imgROS.saveGTTrajectoryAndPositions("GTDcamTrajectory.txt", "GTDcamPosition.txt");
#endif
    std::cout << "Trajectory Saved!" << std::endl;
    exit(SIGINT);

    return 0;
}

void GetImagesROS::currGoalCallBack(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    if ( !onTrack )
    {
        Eigen::Quaterniond q(msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z);
        Eigen::Vector3d t(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
        prevGoal.block<3,3>(0,0) = q.toRotationMatrix();
        prevGoal.block<3,1>(0,3) = t;
        gotPrevGoal = true;
    }
}

#if ATBESTPOSE

void GetImagesROS::aprilTagCallBack(const DC_VSLAM::AprilTagDetectionArray::ConstPtr& msg)
{
    if ( msg->detections.size() == 0 )
    {
        ATFoundCount = 0;
        if ( ATFound )
            ATLostCount++;
        if ( ATLostCount > 10 )
        {
            std::cout << "Starting AprilTag Detection Again!" << std::endl;

            ATFound = false;
            ATLostCount = 0;
        }
        return;
    }
    if (ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED && pathsucRep)
    {
        pathSuccess = true;
        onTrack = false;
        ATFound = false;
        ATFoundCount = 0;
        pathsucRep = false;
    }
    else if ( ac.getState() != actionlib::SimpleClientGoalState::SUCCEEDED )
        pathsucRep = true;

    if ( voSLAM->map->aprilTagDetected || onTrack )
        return;
    if ( gotPrevGoal && poseOptAT && !voSLAM->map->aprilTagDetected )
    {
        std::cout << "Publishing previous Goal!" << std::endl;
        pubGoalAT(msg->header, prevGoal);
        gotPrevGoal = false;
        poseOptAT = false;
    }
    if ( ATFound )
        return;
    ATFoundCount++;
    if ( ATFoundCount < 3 )
        return;
    ATFound = true;
    std::cout << "AprilTag Detected!" << std::endl;
    const geometry_msgs::Point& tra = msg->detections[0].pose.pose.pose.position;
    const geometry_msgs::Quaternion& quat = msg->detections[0].pose.pose.pose.orientation;

    Eigen::Quaterniond qTag(quat.w, quat.x, quat.y, quat.z);
    Eigen::Vector3d tTag(tra.x, tra.y, tra.z);

    Eigen::Matrix4d Tc2_tag = Eigen::Matrix4d::Identity();
    Tc2_tag.block<3,3>(0,0) = qTag.toRotationMatrix();
    Tc2_tag.block<3,1>(0,3) = tTag;
    if ( !tagPoseFilled )
    {
        tagPoseW = T_bf_c1 * voSLAM->mZedCamera->cameraPose.pose * T_c1_AT * Tc2_tag;
        optPose = tagPoseW * T_tag_b;
        tagPoseFilled = true;
        std::cout << "Path to best Pose for AprilTag Detection!" << std::endl;
        onTrack = true;
        pubGoalAT(msg->header, optPose);
        return;
    }
    if ( !first && !pathSuccess )
    {
        std::cout << "Path to best Pose for AprilTag Detection!" << std::endl;
        onTrack = true;
        pubGoalAT(msg->header, optPose);
        return;
    }
    if ( pathSuccess && !onTrack )
    {
        std::cout << "Path Success! " << std::endl;
        if ( first )
        {
            tagPoseW = T_bf_c1 * voSLAM->mZedCamera->cameraPose.pose * T_c1_AT * Tc2_tag;
            optPose = tagPoseW * T_tag_b;
            tagPose = T_bf_c1.inverse() * tagPoseW;
            first = false;
        }
        else
        {
            voSLAM->map->LCPose = tagPose * Tc2_tag.inverse() * T_c1_AT.inverse();
            voSLAM->map->aprilTagDetected = true;
        }
        pathSuccess = false;
        poseOptAT = true;
    }
}

#else

void GetImagesROS::aprilTagCallBack(const DC_VSLAM::AprilTagDetectionArray::ConstPtr& msg)
{
    if ( voSLAM->map->aprilTagDetected )
        return;
    if ( msg->detections.size() == 0 )
    {
        if ( ATFound )
            ATLostCount++;
        if ( ATLostCount > 40 )
        {
            std::cout << "Starting AprilTag Detection Again!" << std::endl;

            ATFound = false;
            ATLostCount = 0;
            ATFoundCount = 0;
        }
        return;
    }
    const geometry_msgs::Point& tra = msg->detections[0].pose.pose.pose.position;
    const geometry_msgs::Quaternion& quat = msg->detections[0].pose.pose.pose.orientation;

    Eigen::Quaterniond qTag(quat.w, quat.x, quat.y, quat.z);
    Eigen::Matrix4d Tc2_tag = Eigen::Matrix4d::Identity();
    Tc2_tag.block<3,3>(0,0) = qTag.normalized().toRotationMatrix();
    Eigen::Vector3d tTag(tra.x, tra.y, tra.z);
    Tc2_tag.block<3,1>(0,3) = tTag;
    if ( ATFound )
        return;
    ATFoundCount++;
    if ( ATFoundCount < 30 )
        return;
    ATFound = true;
    std::cout << "AprilTag Detected!" << std::endl;
    if ( !tagPoseFilled )
    {
        tagPose = voSLAM->mZedCamera->cameraPose.pose * T_c1_AT * Tc2_tag;
        tagPoseFilled = true;
        return;
    }
    std::cout << "AprilTag Detected again.. " << std::endl;
    voSLAM->map->LCPose = tagPose * Tc2_tag.inverse() * T_c1_AT.inverse();
    voSLAM->map->aprilTagDetected = true;

}

#endif

void GetImagesROS::pubGoalAT(const std_msgs::Header& _header, const Eigen::Matrix4d& optPose)
{
    Eigen::Quaterniond q(optPose.block<3,3>(0,0));
    Eigen::Vector3d t = optPose.block<3,1>(0,3);

    move_base_msgs::MoveBaseGoal goal;
    goal.target_pose.header.frame_id = "map";
    goal.target_pose.header.stamp = _header.stamp;
    goal.target_pose.pose.position.x = t.x();
    goal.target_pose.pose.position.y = t.y();
    goal.target_pose.pose.orientation.w = q.w();
    goal.target_pose.pose.orientation.x = q.x();
    goal.target_pose.pose.orientation.y = q.y();
    goal.target_pose.pose.orientation.z = q.z();

    ac.sendGoal(goal);

}

void GetImagesROS::saveGTTrajectoryAndPositions(const std::string& filepath, const std::string& filepathPosition)
{
    std::ofstream datafile(filepath);
    std::ofstream datafilePos(filepathPosition);
    Eigen::Quaterniond& q = gtQuaternions[0];
    Eigen::Matrix4d startPose = Eigen::Matrix4d::Identity();
    Eigen::Vector3d baseToCam(-0.06,0.0,0.15);
    startPose.block<3,3>(0,0) = q.toRotationMatrix();
    startPose.block<3,1>(0,3) = startPose.block<3,3>(0,0) 
    * baseToCam + gtPositions[0];
    Eigen::Matrix4d startPoseInv = (startPose).inverse();
    for ( size_t i{0}, end{gtPositions.size()}; i < end; i ++)
    {
        Eigen::Quaterniond& q = gtQuaternions[i];
        Eigen::Matrix4d Pose = Eigen::Matrix4d::Identity();
        Pose.block<3,3>(0,0) = q.toRotationMatrix();
        Pose.block<3,1>(0,3) = q.toRotationMatrix() * baseToCam +  gtPositions[i];
        Eigen::Matrix4d PoseTT =  startPoseInv * Pose;

        Eigen::Matrix4d PoseT = PoseTT.transpose();
        for (int32_t j{0}; j < 12; j ++)
        {
            if ( j == 0 )
                datafile << PoseT(j);
            else
                datafile << " " << PoseT(j);
            if ( j == 3 || j == 7 || j == 11 )
                datafilePos << PoseT(j) << " ";
        }
        datafile << '\n';
        datafilePos << '\n';
    }
    datafile.close();
    datafilePos.close();

}

void GetImagesROS::getImages(const sensor_msgs::ImageConstPtr& msgLeft,const sensor_msgs::ImageConstPtr& msgRight, const sensor_msgs::ImageConstPtr& msgLeftB,const sensor_msgs::ImageConstPtr& msgRightB)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrLeft;
    try
    {
        cv_ptrLeft = cv_bridge::toCvShare(msgLeft);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrRight;
    try
    {
        cv_ptrRight = cv_bridge::toCvShare(msgRight);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrLeftB;
    try
    {
        cv_ptrLeftB = cv_bridge::toCvShare(msgLeftB);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrRightB;
    try
    {
        cv_ptrRightB = cv_bridge::toCvShare(msgRightB);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat imLeft, imRight;
    cv::Mat imLeftB, imRightB;
    if( !voSLAM->mZedCamera->rectified )
    {
        cv::remap(cv_ptrLeft->image,imLeft, rectMap[0][0],rectMap[0][1],cv::INTER_LINEAR);
        cv::remap(cv_ptrRight->image,imRight,rectMap[1][0],rectMap[1][1],cv::INTER_LINEAR);
    }
    if( !voSLAM->mZedCameraB->rectified )
    {
        cv::remap(cv_ptrLeftB->image,imLeftB, rectMapB[0][0],rectMapB[0][1],cv::INTER_LINEAR);
        cv::remap(cv_ptrRightB->image,imRightB,rectMapB[1][0],rectMapB[1][1],cv::INTER_LINEAR);
    }
    if ( cv_ptrLeft->image.channels() == 1 )
    {
        cv::cvtColor(cv_ptrLeft->image, imLeft, CV_GRAY2BGR);
        cv::cvtColor(cv_ptrRight->image, imRight, CV_GRAY2BGR);
    }
    else
    {
        imLeft = cv_ptrLeft->image;
        imRight = cv_ptrRight->image;
    }
    if ( cv_ptrLeftB->image.channels() == 1 )
    {
        cv::cvtColor(cv_ptrLeftB->image, imLeftB, CV_GRAY2BGR);
        cv::cvtColor(cv_ptrRightB->image, imRightB, CV_GRAY2BGR);
    }
    else
    {
        imLeftB = cv_ptrLeftB->image;
        imRightB = cv_ptrRightB->image;
    }
    voSLAM->trackNewImageMutli(imLeft, imRight, imLeftB, imRightB, frameNumb);
    frameNumb ++;
#if PUBPOINTCLOUD
    pcl::PointCloud<pcl::PointXYZRGB> cloud_;
    const Eigen::Vector3d camPos = T_w_c1.block<3,3>(0,0) * voSLAM->mZedCamera->cameraPose.pose.block<3,1>(0,3);
    const double offset {0.02f};
    const double minH = camPos(2,0) + T_w_c1(2,3) + offset;
    const double maxH = camPos(2,0) + offset;
    const std::vector<DC_VSLAM::MapPoint*>& actMPs = voSLAM->map->activeMapPoints;
    for (size_t i {0}, end{actMPs.size()}; i < end; i ++)
    {
        const DC_VSLAM::MapPoint* mp = actMPs[i];
        if ( !mp || mp->GetIsOutlier() || mp->unMCnt != 0 || (mp->kFMatches.size() < 2 && mp->kFMatchesB.size() < 2) )
            continue;
        Eigen::Vector3d pt = mp->getWordPose3d();
        pt = T_w_c1.block<3,3>(0,0) * pt;
        std::cout << "point : " << pt << std::endl;
        if ( pt(2) < minH || pt(2) > maxH )
            continue;
        pcl::PointXYZRGB ptpcl;
        ptpcl.x = pt(0);
        ptpcl.y = pt(1);
        ptpcl.z = pt(2);
        cloud_.points.push_back(ptpcl);
    }
    const std::vector<DC_VSLAM::MapPoint*>& actMPsB = voSLAM->map->activeMapPointsB;
    for (size_t i {0}, end{actMPsB.size()}; i < end; i ++)
    {
        const DC_VSLAM::MapPoint* mp = actMPsB[i];
        if ( !mp || mp->GetIsOutlier() || mp->unMCnt != 0 || (mp->kFMatches.size() < 2 && mp->kFMatchesB.size() < 2) )
            continue;
        Eigen::Vector3d pt = mp->getWordPose3d();
        pt = T_w_c1.block<3,3>(0,0) * pt;
        // std::cout << "point : " << pt << std::endl;
        if ( pt(2) < minH || pt(2) > maxH )
            continue;
        pcl::PointXYZRGB ptpcl;
        ptpcl.x = pt(0);
        ptpcl.y = pt(1);
        ptpcl.z = pt(2);
        cloud_.points.push_back(ptpcl);
    }
    sensor_msgs::PointCloud2 pc2_msg_;
    pcl::toROSMsg(cloud_, pc2_msg_);
    pc2_msg_.header.frame_id = "map";
    pc2_msg_.header.stamp = msgLeft->header.stamp;
    pc_pub.publish(pc2_msg_);
    publishOdom(msgLeft->header);
#endif
}

void GetImagesROS::getImages(const sensor_msgs::ImageConstPtr& msgLeft,const sensor_msgs::ImageConstPtr& msgRight, const sensor_msgs::ImageConstPtr& msgLeftB,const sensor_msgs::ImageConstPtr& msgRightB,const nav_msgs::OdometryConstPtr& msgGT)
{

    gtPositions.emplace_back(-msgGT->pose.pose.position.y, msgGT->pose.pose.position.z, msgGT->pose.pose.position.x);
    gtQuaternions.emplace_back(-msgGT->pose.pose.orientation.w, msgGT->pose.pose.orientation.y, msgGT->pose.pose.orientation.z, -msgGT->pose.pose.orientation.x);
    
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrLeft;
    try
    {
        cv_ptrLeft = cv_bridge::toCvShare(msgLeft);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrRight;
    try
    {
        cv_ptrRight = cv_bridge::toCvShare(msgRight);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrLeftB;
    try
    {
        cv_ptrLeftB = cv_bridge::toCvShare(msgLeftB);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrRightB;
    try
    {
        cv_ptrRightB = cv_bridge::toCvShare(msgRightB);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat imLeft, imRight;
    cv::Mat imLeftB, imRightB;
    if( !voSLAM->mZedCamera->rectified )
    {
        cv::remap(cv_ptrLeft->image,imLeft, rectMap[0][0],rectMap[0][1],cv::INTER_LINEAR);
        cv::remap(cv_ptrRight->image,imRight,rectMap[1][0],rectMap[1][1],cv::INTER_LINEAR);
    }
    if( !voSLAM->mZedCameraB->rectified )
    {
        cv::remap(cv_ptrLeftB->image,imLeftB, rectMapB[0][0],rectMapB[0][1],cv::INTER_LINEAR);
        cv::remap(cv_ptrRightB->image,imRightB,rectMapB[1][0],rectMapB[1][1],cv::INTER_LINEAR);
    }
    if ( cv_ptrLeft->image.channels() == 1 )
    {
        cv::cvtColor(cv_ptrLeft->image, imLeft, CV_GRAY2BGR);
        cv::cvtColor(cv_ptrRight->image, imRight, CV_GRAY2BGR);
    }
    else
    {
        imLeft = cv_ptrLeft->image;
        imRight = cv_ptrRight->image;
    }
    if ( cv_ptrLeftB->image.channels() == 1 )
    {
        cv::cvtColor(cv_ptrLeftB->image, imLeftB, CV_GRAY2BGR);
        cv::cvtColor(cv_ptrRightB->image, imRightB, CV_GRAY2BGR);
    }
    else
    {
        imLeftB = cv_ptrLeftB->image;
        imRightB = cv_ptrRightB->image;
    }
    voSLAM->trackNewImageMutli(imLeft, imRight, imLeftB, imRightB, frameNumb);
    frameNumb ++;
#if PUBPOINTCLOUD
    pcl::PointCloud<pcl::PointXYZRGB> cloud_;
    const Eigen::Vector3d camPos = T_w_c1.block<3,3>(0,0) * voSLAM->mZedCamera->cameraPose.pose.block<3,1>(0,3);
    const double offset {0.02f};
    const double minH = camPos(2,0) + T_w_c1(2,3) + offset;
    const double maxH = camPos(2,0) + offset;
    const std::vector<DC_VSLAM::MapPoint*>& actMPs = voSLAM->map->activeMapPoints;
    for (size_t i {0}, end{actMPs.size()}; i < end; i ++)
    {
        const DC_VSLAM::MapPoint* mp = actMPs[i];
        if ( !mp || mp->GetIsOutlier() || mp->unMCnt != 0 || (mp->kFMatches.size() < 2 && mp->kFMatchesB.size() < 2) )
            continue;
        Eigen::Vector3d pt = mp->getWordPose3d();
        pt = T_w_c1.block<3,3>(0,0) * pt;
        if ( pt(2) < minH || pt(2) > maxH )
            continue;
        pcl::PointXYZRGB ptpcl;
        ptpcl.x = pt(0);
        ptpcl.y = pt(1);
        ptpcl.z = pt(2);
        cloud_.points.push_back(ptpcl);
    }
    const std::vector<DC_VSLAM::MapPoint*>& actMPsB = voSLAM->map->activeMapPointsB;
    for (size_t i {0}, end{actMPsB.size()}; i < end; i ++)
    {
        const DC_VSLAM::MapPoint* mp = actMPsB[i];
        if ( !mp || mp->GetIsOutlier() || mp->unMCnt != 0 || (mp->kFMatches.size() < 2 && mp->kFMatchesB.size() < 2) )
            continue;
        Eigen::Vector3d pt = mp->getWordPose3d();
        pt = T_w_c1.block<3,3>(0,0) * pt;
        if ( pt(2) < minH || pt(2) > maxH )
            continue;
        pcl::PointXYZRGB ptpcl;
        ptpcl.x = pt(0);
        ptpcl.y = pt(1);
        ptpcl.z = pt(2);
        cloud_.points.push_back(ptpcl);
    }
    sensor_msgs::PointCloud2 pc2_msg_;
    pcl::toROSMsg(cloud_, pc2_msg_);
    pc2_msg_.header.frame_id = "odom";
    pc2_msg_.header.stamp = msgLeft->header.stamp;
    pc_pub.publish(pc2_msg_);
    publishOdom(msgLeft->header);
#endif
}

void GetImagesROS::publishOdom(const std_msgs::Header& _header)
{
    Eigen::Matrix4d startPose = Eigen::Matrix4d::Identity();
    // tf on the cordinate system of the camera. so -0.15 on z(backwards from camera)
    Eigen::Vector3d camToBase(0.06,0.0,-0.15);
    Eigen::Matrix4d camPose = voSLAM->mZedCamera->cameraPose.pose;
    startPose.block<3,1>(0,3) = startPose.block<3,3>(0,0) 
    * camToBase;
    camPose.block<3,1>(0,3) = camPose.block<3,3>(0,0) * camToBase + camPose.block<3,1>(0,3);
    camPose = startPose.inverse() * camPose;
    const Eigen::Quaterniond q(camPose.block<3,3>(0,0));
    Eigen::Vector3d tra = camPose.block<3,1>(0,3);
    tra = T_w_c1.block<3,3>(0,0) * tra;
    geometry_msgs::TransformStamped odom_trans;
    odom_trans.header.stamp = _header.stamp;
    odom_trans.header.frame_id = "odom";
    odom_trans.child_frame_id = "base_footprint";

    odom_trans.transform.translation.x = tra(0);
    odom_trans.transform.translation.y = tra(1);
    odom_trans.transform.translation.z = tra(2);
    
    odom_trans.transform.rotation.w = -q.w();
    odom_trans.transform.rotation.x = -q.z();
    odom_trans.transform.rotation.y = q.x();
    odom_trans.transform.rotation.z = q.y();

    odom_broadcaster.sendTransform(odom_trans);
    
    nav_msgs::Odometry odom;
    odom.header.stamp = _header.stamp;
    odom.header.frame_id = "odom";

    //set the position
    odom.pose.pose.position.x = tra(0);
    odom.pose.pose.position.y = tra(1);
    odom.pose.pose.position.z = tra(2);

    odom.pose.pose.orientation.w = -q.w();
    odom.pose.pose.orientation.x = -q.z();
    odom.pose.pose.orientation.y = q.x();
    odom.pose.pose.orientation.z = q.y();

    //set the velocity
    odom.child_frame_id = "base_footprint";
    odom.twist.twist.linear.x = 0;
    odom.twist.twist.linear.y = 0;
    odom.twist.twist.angular.z = 0;

    odom_pub.publish(odom);
}
