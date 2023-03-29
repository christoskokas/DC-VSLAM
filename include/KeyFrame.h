#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "Settings.h"
#include "Camera.h"
#include "FeatureExtractor.h"
#include "Map.h"
#include "opencv2/core.hpp"


namespace DC_VSLAM
{

class Map;
class MapPoint;

class KeyFrame
{
    private:

    public:
        double fx,fy,cx,cy;
        double fxb,fyb,cxb,cyb;
        CameraPose pose;
        Eigen::Matrix4d extr;
        Eigen::Matrix4d extrB;
        Eigen::Matrix4d TCamToCam;
        Eigen::Matrix4d backPose;
        Eigen::Matrix4d backPoseInv;
        cv::Mat leftIm, rightIm;
        cv::Mat rLeftIm;
        std::vector<int> unMatchedF;
        std::vector<int> unMatchedFR;
        std::vector<int> unMatchedFB;
        std::vector<int> unMatchedFRB;
        std::vector<float> scaleFactor;
        std::vector < float > sigmaFactor;
        std::vector < float > InvSigmaFactor;
        std::unordered_map<KeyFrame*, int> weightsKF;
        std::vector<std::pair<int,KeyFrame*>> sortedKFWeights;
        float logScale;
        int nScaleLev;

        int LBAID {-1};
        int LCID {-1};

        bool LCCand {false};

        TrackedKeys keys, keysB;
        Eigen::MatrixXd homoPoints3D;
        const unsigned long numb;
        const int frameIdx;
        int nKeysTracked {0};
        bool visualize {true};
        std::vector<MapPoint*> localMapPoints;
        std::vector<MapPoint*> localMapPointsR;
        std::vector<MapPoint*> localMapPointsB;
        std::vector<MapPoint*> localMapPointsRB;
        KeyFrame* prevKF = nullptr;
        KeyFrame* nextKF = nullptr;
        bool active {true};
        bool keyF {false};
        bool LBA {false};
        bool fixed {false};
        bool backCam {false};

        void updatePose(const Eigen::Matrix4d& keyPose);

        // Create Function that updates connections
        void calcConnections();

        // Create Function that updates connections

        void setBackPose(const Eigen::Matrix4d& _backPose);
        void eraseMPConnection(const int mpPos);
        void eraseMPConnectionB(const int mpPos);
        void eraseMPConnection(const std::pair<int,int>& mpPos);
        void eraseMPConnectionB(const std::pair<int,int>& mpPos);
        void eraseMPConnectionR(const int mpPos);
        void eraseMPConnectionRB(const int mpPos);
        KeyFrame(Eigen::Matrix4d _pose, cv::Mat& _leftIm, cv::Mat& rLIm, const int _numb, const int _frameIdx);
        KeyFrame(const Eigen::Matrix4d& _refPose, const Eigen::Matrix4d& realPose, cv::Mat& _leftIm, cv::Mat& rLIm, const int _numb, const int _frameIdx);
        KeyFrame(const Zed_Camera* _zedCam, const Eigen::Matrix4d& _refPose, const Eigen::Matrix4d& realPose, cv::Mat& _leftIm, cv::Mat& rLIm, const int _numb, const int _frameIdx);
        KeyFrame(const Zed_Camera* _zedCam, const Zed_Camera* _zedCamB, const Eigen::Matrix4d& _refPose, const Eigen::Matrix4d& realPose, cv::Mat& _leftIm, cv::Mat& rLIm, const int _numb, const int _frameIdx);
        Eigen::Vector4d getWorldPosition(int idx);
        void getConnectedKFs(std::vector<KeyFrame*>& activeKF, const int N);
        void getConnectedKFsLC(const Map* map, std::vector<KeyFrame*>& activeKF);

        Eigen::Matrix4d getPose();
};

} // namespace DC_VSLAM

#endif // KEYFRAME_H