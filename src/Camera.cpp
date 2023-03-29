#include "Camera.h"



namespace DC_VSLAM
{

CameraPose::CameraPose(Eigen::Matrix4d _pose, std::chrono::time_point<std::chrono::high_resolution_clock> _timestamp) : pose(_pose), timestamp(_timestamp)
{}

void CameraPose::setPose(const Eigen::Matrix4d& poseT)
{
    pose = poseT;
    poseInverse = poseT.inverse();
    timestamp = std::chrono::high_resolution_clock::now();
}

Eigen::Matrix4d CameraPose::getPose() const
{
    return pose;
}

Eigen::Matrix4d CameraPose::getInvPose() const
{
    return poseInverse;
}

void CameraPose::setPose(Eigen::Matrix4d& _refPose, Eigen::Matrix4d& _keyPose)
{
    refPose = _refPose;
    Eigen::Matrix4d truePose = _keyPose * refPose;
    setPose(truePose);
}

void CameraPose::changePose(const Eigen::Matrix4d& _keyPose)
{
    Eigen::Matrix4d truePose = _keyPose * refPose;
    setPose(truePose);
}

void CameraPose::setInvPose(const Eigen::Matrix4d poseT)
{
    poseInverse = poseT;
    pose = poseT.inverse();
}

Zed_Camera::Zed_Camera(ConfigFile* yamlFile)
{
    confFile = yamlFile;
    rectified = confFile->getValue<bool>("rectified");
    numOfFrames = confFile->getValue<int>("numOfFrames"); 
    setCameraValues("Camera");

}

Zed_Camera::Zed_Camera(ConfigFile* yamlFile, bool backCamera)
{
    confFile = yamlFile;
    numOfFrames = confFile->getValue<int>("numOfFrames");
    if ( !backCamera )
    {
        rectified = confFile->getValue<bool>("rectified");
        setCameraValues("Camera");
    }
    else
    {
        rectified = confFile->getValue<bool>("rectifiedB");
        setCameraValues("CameraB");
    }
    setBackCameraT(backCamera);

}

void Zed_Camera::setBackCameraT(const bool backCamera)
{
    std::vector<float> Tc1_c2 = confFile->getValue<std::vector<float>>("Multi", "T_c1_c2", "data");

    Eigen::Matrix4d transfC1C2;
    transfC1C2 << Tc1_c2[0],Tc1_c2[1],Tc1_c2[2],Tc1_c2[3],Tc1_c2[4],Tc1_c2[5],Tc1_c2[6],Tc1_c2[7],Tc1_c2[8],Tc1_c2[9],Tc1_c2[10],Tc1_c2[11],Tc1_c2[12],Tc1_c2[13],Tc1_c2[14],Tc1_c2[15];
    Eigen::Matrix4d transfC1C2inv = transfC1C2.inverse();

    if ( backCamera )
    {
        cameraPose.setPose(transfC1C2);
        TCamToCam = transfC1C2inv;
        TCamToCamInv = transfC1C2;
    }
    else
    {
        TCamToCam = transfC1C2;
        TCamToCamInv = transfC1C2inv;
    }
}

void Zed_Camera::setCameraValues(const std::string& camPath)
{
    mWidth = confFile->getValue<int>(camPath,"width");
    mHeight = confFile->getValue<int>(camPath,"height");
    mFps = confFile->getValue<float>(camPath,"fps");
    if ( rectified )
    {
        cameraLeft.setIntrinsicValuesR(camPath + "_l",confFile);
        cameraRight.setIntrinsicValuesR(camPath + "_r",confFile);
    }
    else
    {
        cameraLeft.setIntrinsicValuesUnR(camPath + "_l",confFile);
        cameraRight.setIntrinsicValuesUnR(camPath + "_r",confFile);
    }
    mBaseline = confFile->getValue<float>(camPath ,"bl");
    extrinsics(0,3) = (double)mBaseline;
}

Zed_Camera::~Zed_Camera()
{

}

Camera::~Camera()
{
    
}

void Camera::setIntrinsicValuesUnR(const std::string& cameraPath, ConfigFile* confFile)
{
    fx = confFile->getValue<double>(cameraPath,"fx");
    fy = confFile->getValue<double>(cameraPath,"fy");
    cx = confFile->getValue<double>(cameraPath,"cx");
    cy = confFile->getValue<double>(cameraPath,"cy");
    k1 = confFile->getValue<double>(cameraPath,"k1");
    k2 = confFile->getValue<double>(cameraPath,"k2");
    p1 = confFile->getValue<double>(cameraPath,"p1");
    p2 = confFile->getValue<double>(cameraPath,"p2");
    k3 = confFile->getValue<double>(cameraPath,"k3");
    std::vector < double > Rt = confFile->getValue<std::vector<double>>(cameraPath,"R","data");
    std::vector < double > Pt = confFile->getValue<std::vector<double>>(cameraPath,"P","data");
    std::vector < double > Dt = confFile->getValue<std::vector<double>>(cameraPath,"D","data");
    std::vector < double > Kt = confFile->getValue<std::vector<double>>(cameraPath,"K","data");

    R = (cv::Mat_<double>(3,3) << Rt[0], Rt[1], Rt[2], Rt[3], Rt[4], Rt[5], Rt[6], Rt[7], Rt[8]);
    P = (cv::Mat_<double>(3,4) << Pt[0], Pt[1], Pt[2], Pt[3], Pt[4], Pt[5], Pt[6], Pt[7], Pt[8], Pt[9], Pt[10], Pt[11]);
    K = (cv::Mat_<double>(3,3) << Kt[0], Kt[1], Kt[2], Kt[3], Kt[4], Kt[5], Kt[6], Kt[7], Kt[8]);
    D = (cv::Mat_<double>(1,5) << Dt[0], Dt[1], Dt[2], Dt[3], Dt[4]);

    intrinsics(0,0) = fx;
    intrinsics(1,1) = fy;
    intrinsics(0,2) = cx;
    intrinsics(1,2) = cy;

}

void Camera::setIntrinsicValuesR(const std::string& cameraPath, ConfigFile* confFile)
{

    fx = confFile->getValue<double>(cameraPath,"fx");
    fy = confFile->getValue<double>(cameraPath,"fy");
    cx = confFile->getValue<double>(cameraPath,"cx");
    cy = confFile->getValue<double>(cameraPath,"cy");
    k1 = 0;
    k2 = 0;
    p1 = 0;
    p2 = 0;
    k3 = 0;

    intrinsics(0,0) = fx;
    intrinsics(1,1) = fy;
    intrinsics(0,2) = cx;
    intrinsics(1,2) = cy;
}

} //namespace DC_VSLAM

