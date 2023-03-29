#ifndef CONVERSIONS_H
#define CONVERSIONS_H

#include <opencv2/calib3d.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/video/tracking.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>



namespace DC_VSLAM
{

struct Converter
{

    static Eigen::Matrix3d convertCVRotToEigen(cv::Mat& Rot)
    {
        cv::Rodrigues(Rot,Rot);
        Eigen::Matrix3d RotEig;
        cv::cv2eigen(Rot, RotEig);
        return RotEig;
    }
    static Eigen::Vector3d convertCVTraToEigen(cv::Mat& Tra)
    {
        Eigen::Vector3d traEig;
        cv::cv2eigen(Tra, traEig);
        return traEig;
    }

    static Eigen::Matrix4d convertRTtoPose(cv::Mat& Rot, cv::Mat& Tra)
    {
        Eigen::Vector3d traEig;
        cv::cv2eigen(Tra, traEig);
        cv::Rodrigues(Rot,Rot);
        Eigen::Matrix3d RotEig;
        cv::cv2eigen(Rot, RotEig);

        Eigen::Matrix4d convPose = Eigen::Matrix4d::Identity();
        convPose.block<3,3>(0,0) = RotEig;
        convPose.block<3,1>(0,3) = traEig;

        return convPose;
    }

    static void convertEigenPoseToMat(const Eigen::Matrix4d& poseToConv, cv::Mat& Rot, cv::Mat& Tra)
    {
        Eigen::Matrix3d RotEig;
        Eigen::Vector3d TraEig;
        RotEig = poseToConv.block<3,3>(0,0);
        TraEig = poseToConv.block<3,1>(0,3);

        cv::eigen2cv(RotEig,Rot);
        cv::eigen2cv(TraEig,Tra);

        cv::Rodrigues(Rot, Rot);
    }

    static Eigen::Matrix<double, 7, 1> Matrix4dToMatrix_7_1(
    const Eigen::Matrix4d& pose) 
    {
        Eigen::Matrix<double, 7, 1> Tcw_7_1;
        Eigen::Matrix3d R;
        R = pose.block<3, 3>(0, 0);
        // Eigen Quaternion coeffs output [x, y, z, w]
        Tcw_7_1.block<3, 1>(0, 0) = pose.block<3, 1>(0, 3);
        Tcw_7_1.block<4, 1>(3, 0) = Eigen::Quaterniond(R).coeffs();
        return Tcw_7_1;
    }

    static Eigen::Matrix4d Matrix_7_1_ToMatrix4d(
    const Eigen::Matrix<double, 7, 1>& Tcw_7_1) 
    {
        Eigen::Quaterniond q(Tcw_7_1[6], Tcw_7_1[3], Tcw_7_1[4], Tcw_7_1[5]);
        Eigen::Matrix3d R = q.normalized().toRotationMatrix();
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose.block<3, 3>(0, 0) = R;
        pose.block<3, 1>(0, 3) = Tcw_7_1.block<3, 1>(0, 0);
        return pose;
    }

};

} // namespace DC_VSLAM


#endif // CONVERSIONS_H