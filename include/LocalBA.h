#ifndef LOCALBA_H
#define LOCALBA_H


#include "Camera.h"
#include "KeyFrame.h"
#include "Map.h"
#include "FeatureExtractor.h"
#include "FeatureMatcher.h"
#include "Conversions.h"
#include "Settings.h"
#include "Optimizer.h"
#include "Eigen/Dense"
#include <fstream>
#include <string>
#include <iostream>
#include <random>

namespace DC_VSLAM
{


class LocalMapper
{
    private:

    public:

        bool stopRequested {false};


        Map* map;

        const float reprjThreshold {7.815f};

        const int actvKFMaxSize {10};
        const int minCount {3};

        Zed_Camera* zedPtr;
        Zed_Camera* zedPtrB;

        FeatureMatcher* fm;
        const double fx,fy,cx,cy;

        LocalMapper(Map* _map, Zed_Camera* _zedPtr, FeatureMatcher* _fm);

        LocalMapper(Map* _map, Zed_Camera* _zedPtr, Zed_Camera* _zedPtrB, FeatureMatcher* _fm);

        // loop closure optimization
        void loopClosureR(std::vector<DC_VSLAM::KeyFrame *>& actKeyF);
        void loopClosureRB(std::vector<DC_VSLAM::KeyFrame *>& actKeyF);

        // loop closure check
        void beginLoopClosure();
        void beginLoopClosureB();

        // insert MPs from localMapPoints for optimization
        void insertMPsForLBA(std::vector<MapPoint*>& localMapPoints, const std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>& localKFs,std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>& fixedKFs, std::unordered_map<MapPoint*, Eigen::Vector3d>& allMapPoints, const unsigned long lastActKF, int& blocks, const bool back);
        void insertMPsForLC(std::vector<MapPoint*>& localMapPoints, const std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>& localKFs, std::unordered_map<MapPoint*, Eigen::Vector3d>& allMapPoints, const unsigned long lastActKF, int& blocks, const bool back);
        
        // triangulate new points from connected keyframes
        void triangulateNewPointsR(std::vector<DC_VSLAM::KeyFrame *>& activeKF);
        void triangulateNewPointsRB(const Zed_Camera* zedCam,std::vector<DC_VSLAM::KeyFrame *>& activeKF, const bool back);
        
        // find all keypoints that have a stereo match for triangulating new points
        void calcAllMpsOfKFROnlyEstB(const Zed_Camera* zedCam, std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs, KeyFrame* lastKF, const int kFsize, std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<float>& maxDistsScale, const bool back);
        void calcAllMpsOfKFROnlyEst(std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs, KeyFrame* lastKF, const int kFsize, std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<float>& maxDistsScale);

        // predict the mps position on the connected KFs
        void predictKeysPosR(const TrackedKeys& keys, const Eigen::Matrix4d& camPose, const Eigen::Matrix4d& camPoseInv, const std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<std::pair<cv::Point2f, cv::Point2f>>& predPoints);
        void predictKeysPosRB(const Zed_Camera* zedCam, const Eigen::Matrix4d& camPose, const Eigen::Matrix4d& camPoseInv, const std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<std::pair<cv::Point2f, cv::Point2f>>& predPoints);

        // local BA optimization
        void localBAR(std::vector<DC_VSLAM::KeyFrame *>& actKeyF);
        void localBARB(std::vector<DC_VSLAM::KeyFrame *>& actKeyF);

        // check the reprojection error between matched keypoints (from triangulation)
        bool checkReprojErrNewR(KeyFrame* lastKF, Eigen::Vector4d& calcVec, std::vector<std::pair<KeyFrame *, std::pair<int, int>>>& matchesOfPoint, const std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& pointsVec);
        bool checkReprojErrNewRB(KeyFrame* lastKF, Eigen::Vector4d& calcVec, std::vector<std::pair<KeyFrame *, std::pair<int, int>>>& matchesOfPoint, const std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& pointsVec, const bool back);

        
        // calculate projection matrices for triangulation
        void calcProjMatricesR(std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>>& projMatrices, std::vector<KeyFrame*>& actKeyF);
        void calcProjMatricesRB(const Zed_Camera* zedCam,std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>>& projMatrices, std::vector<KeyFrame*>& actKeyF,const bool back);

        // process matches to find the optimized 3D position of the mappoint
        void processMatchesR(std::vector<std::pair<DC_VSLAM::KeyFrame *, std::pair<int, int>>>& matchesOfPoint, std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>>& allProjMatrices, std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& points);
        void processMatchesRB(std::vector<std::pair<DC_VSLAM::KeyFrame *, std::pair<int, int>>>& matchesOfPoint, std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>>& allProjMatrices, std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& points, const bool back);

        // add optimized mappoints to vector for insertion to the map
        void addMultiViewMapPointsR(const Eigen::Vector4d& posW, const std::vector<std::pair<DC_VSLAM::KeyFrame *, std::pair<int, int>>>& matchesOfPoint, std::vector<MapPoint*>& pointsToAdd, KeyFrame* lastKF, const size_t& mpPos);
        void addMultiViewMapPointsRB(const Eigen::Vector4d& posW, const std::vector<std::pair<DC_VSLAM::KeyFrame *, std::pair<int, int>>>& matchesOfPoint, std::vector<MapPoint*>& pointsToAdd, KeyFrame* lastKF, const size_t& mpPos, const bool back);

        // add the optimized mappoints to the map
        void addNewMapPoints(KeyFrame* lastKF, std::vector<MapPoint*>& pointsToAdd, std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs);
        void addNewMapPointsB(KeyFrame* lastKF, std::vector<MapPoint*>& pointsToAdd, std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs, const bool back);

        
        // check if mappoint is outlier
        bool checkOutlier(const Eigen::Matrix3d& K, const Eigen::Vector2d& obs, const Eigen::Vector3d posW,const Eigen::Vector3d& tcw, const Eigen::Quaterniond& qcw, const float thresh);
        bool checkOutlierR(const Eigen::Matrix3d& K, const Eigen::Matrix3d& qc1c2, const Eigen::Matrix<double,3,1>& tc1c2, const Eigen::Vector2d& obs, const Eigen::Vector3d posW,const Eigen::Vector3d& tcw, const Eigen::Quaterniond& qcw, const float thresh);

        // local BA check
        void beginLocalMapping();
        void beginLocalMappingB();
        void triangulateCeresNew(Eigen::Vector3d& p3d, const std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, const std::vector<Eigen::Vector2d>& obs, const Eigen::Matrix4d& lastKFPose, bool first);

};



} // namespace DC_VSLAM


#endif // LOCALBA_H