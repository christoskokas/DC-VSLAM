#ifndef FEATURETRACKER_H
#define FEATURETRACKER_H

#include "Camera.h"
#include "KeyFrame.h"
#include "Map.h"
#include "FeatureExtractor.h"
#include "FeatureMatcher.h"
#include "Conversions.h"
#include "Settings.h"
#include "Optimizer.h"
#include <fstream>
#include <string>
#include <iostream>
#include <random>

namespace DC_VSLAM
{

class ImageData
{
    private:

    public:
        cv::Mat im, rIm;
};

class FeatureTracker
{
    private :

        std::vector<KeyFrame> keyframes;


        KeyFrame* latestKF = nullptr;
        Eigen::Matrix4d lastKFPoseInv = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d poseEst = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d predNPose = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d predNPoseInv = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d predNPoseRef = Eigen::Matrix4d::Identity();

        const int actvKFMaxSize {10};
        const int minNStereo {80};
        const int maxAddedStereo {100};
        const int minInliers {50};

        float precCheckMatches {0.9f};

        int lastKFTrackedNumb {0};

        double fxb,fyb,cxb,cyb;

        const int keyFrameCountEnd {5};
        int insertKeyFrameCount {0};
        int curFrame {0};
        int curFrameNumb {-1};

        ImageData pLIm, pRIm, lIm, rIm;
        Zed_Camera* zedPtr;
        Zed_Camera* zedPtrB = nullptr;
        Map* map;
        FeatureMatcher fm;
        FeatureMatcher fmB;
        const double fx,fy,cx,cy;
        // FeatureExtractor fe;
        FeatureExtractor* feLeft;
        FeatureExtractor* feRight;
        FeatureExtractor* feLeftB;
        FeatureExtractor* feRightB;

        std::vector<MapPoint*>& activeMapPoints;
        std::vector<MapPoint*>& activeMapPointsB;
        std::vector<KeyFrame*>& allFrames;

    public :

        FeatureTracker(Zed_Camera* _zedPtr, FeatureExtractor* _feLeft, FeatureExtractor* _feRight, Map* _map);
        FeatureTracker(Zed_Camera* _zedPtr, Zed_Camera* _zedPtrB, FeatureExtractor* _feLeft, FeatureExtractor* _feRight, FeatureExtractor* _feLeftB, FeatureExtractor* _feRightB, Map* _map);

        // main tracking function
        void TrackImageT(const cv::Mat& leftRect, const cv::Mat& rightRect, const int frameNumb);
        void TrackImageTB(const cv::Mat& leftRect, const cv::Mat& rightRect, const cv::Mat& leftRectB, const cv::Mat& rightRectB, const int frameNumb);

        // extract orb features
        void extractORBStereoMatchR(cv::Mat& leftIm, cv::Mat& rightIm, TrackedKeys& keysLeft);
        void extractORBStereoMatchRB(const Zed_Camera* zedCam, cv::Mat& leftIm, cv::Mat& rightIm, FeatureExtractor* feLeft, FeatureExtractor* feRight, FeatureMatcher& fm, TrackedKeys& keysLeft);

        // Initialize map with 3D mappoints
        void initializeMapR(TrackedKeys& keysLeft);
        void initializeMapRB(TrackedKeys& keysLeft, TrackedKeys& keysLeftB);

        // set 3D mappoints as outliers
        void setActiveOutliers(std::vector<MapPoint*>& activeMPs, std::vector<bool>& MPsOutliers, std::vector<std::pair<int,int>>& matchesIdxs);

        // remove mappoints that are out of frame
        void removeOutOfFrameMPsR(const Eigen::Matrix4d& currCamPose, const Eigen::Matrix4d& predNPose, std::vector<MapPoint*>& activeMapPoints);
        void removeOutOfFrameMPsRB(const Zed_Camera* zedCam, const Eigen::Matrix4d& predNPose, std::vector<MapPoint*>& activeMapPoints);

        // 3d world coords to frame coords
        bool worldToFrameRTrack(MapPoint* mp, const bool right, const Eigen::Matrix4d& predPoseInv, const Eigen::Matrix4d& tempPose);
        bool worldToFrameRTrackB(MapPoint* mp, const Zed_Camera* zedCam, const bool right, const Eigen::Matrix4d& predPoseInv);

        // pose estimation ( Ceres Solver )
        std::pair<int,int> estimatePoseCeresR(std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<std::pair<int,int>>& matchesIdxs, Eigen::Matrix4d& estimPose, std::vector<bool>& MPsOutliers, const bool first);
        std::pair<std::pair<int,int>, std::pair<int,int>> estimatePoseCeresRB(std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<std::pair<int,int>>& matchesIdxs, std::vector<bool>& MPsOutliers, std::vector<MapPoint*>& activeMapPointsB, TrackedKeys& keysLeftB, std::vector<std::pair<int,int>>& matchesIdxsB, std::vector<bool>& MPsOutliersB, Eigen::Matrix4d& estimPose);

        // check for outliers after pose estimation
        int findOutliersR(const Eigen::Matrix4d& estimatedP, std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<std::pair<int,int>>& matchesIdxs, const double thres, std::vector<bool>& MPsOutliers, const std::vector<float>& weights, int& nInliers);
        int findOutliersRB(const Zed_Camera* zedCam, const Eigen::Matrix4d& estimatedP, std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<std::pair<int,int>>& matchesIdxs, const double thres, std::vector<bool>& MPsOutliers, int& nInliers);

        // predict position of 3d mappoints with predicted camera pose
        void newPredictMPs(const Eigen::Matrix4d& currCamPose, const Eigen::Matrix4d& predNPose, std::vector<MapPoint*>& activeMapPoints, std::vector<int>& matchedIdxsL, std::vector<int>& matchedIdxsR, std::vector<std::pair<int,int>>& matchesIdxs, std::vector<bool> &MPsOutliers);
        void newPredictMPsB(const Zed_Camera* zedCam, const Eigen::Matrix4d& predNPose, std::vector<MapPoint*>& activeMapPoints, std::vector<int>& matchedIdxsL, std::vector<int>& matchedIdxsR, std::vector<std::pair<int,int>>& matchesIdxs, std::vector<bool> &MPsOutliers);

        // insert KF if needed
        void insertKeyFrameR(TrackedKeys& keysLeft, std::vector<int>& matchedIdxsL, std::vector<std::pair<int,int>>& matchesIdxs, const int nStereo, const Eigen::Matrix4d& estimPose, std::vector<bool>& MPsOutliers, cv::Mat& leftIm, cv::Mat& rleftIm);
        void insertKeyFrameRB(TrackedKeys& keysLeft, std::vector<int>& matchedIdxsL, std::vector<std::pair<int,int>>& matchesIdxs, std::vector<bool>& MPsOutliers, TrackedKeys& keysLeftB, std::vector<int>& matchedIdxsLB, std::vector<std::pair<int,int>>& matchesIdxsB, std::vector<bool>& MPsOutliersB, const int nStereo, const int nStereoB, const Eigen::Matrix4d& estimPose, cv::Mat& leftIm, cv::Mat& rleftIm);

        // check 2d Error
        bool check2dError(Eigen::Vector4d& p4d, const cv::Point2f& obs, const double thres, const double weight);
        bool check2dErrorB(const Zed_Camera* zedCam, Eigen::Vector4d& p4d, const cv::Point2f& obs, const double thres, const double weight);

        // change camera poses after either Local BA or Global BA
        void changePosesLCA(const int endIdx);
        void changePosesLCAB(const int endIdx);

        // publish camera pose
        void publishPoseNew();
        void publishPoseNewB();

        // add frame if not KF
        void addFrame(const Eigen::Matrix4d& estimPose);

        // assign features to grids for faster matching
        void assignKeysToGrids(TrackedKeys& keysLeft, std::vector<cv::KeyPoint>& keypoints,std::vector<std::vector<std::vector<int>>>& keyGrid, const int width, const int height);

        // draw tracked keypoints ( TODO move this to the visual thread )
        void drawKeys(const char* com, cv::Mat& im, std::vector<cv::KeyPoint>& keys, std::vector<bool>& close);

};



} // namespace DC_VSLAM


#endif // FEATURETRACKER_H