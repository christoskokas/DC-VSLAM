#ifndef MAP_H
#define MAP_H

#include "Camera.h"
#include "KeyFrame.h"
#include "FeatureMatcher.h"
#include "Settings.h"
#include "Optimizer.h"
#include <fstream>
#include <string>
#include <iostream>
#include <random>
#include <unordered_map>

namespace DC_VSLAM
{

class KeyFrame;

class MapPoint
{
    private:

    public:

        Eigen::Vector4d wp;
        Eigen::Vector3d wp3d;
        int unMCnt {0};
        std::vector<cv::KeyPoint> obs;
        cv::KeyPoint lastObsL;
        cv::KeyPoint lastObsR;
        KeyFrame* lastObsKF;
        cv::Mat desc;
        std::unordered_map<KeyFrame*, std::pair<int,int>> kFMatches;
        std::unordered_map<KeyFrame*, std::pair<int,int>> kFMatchesB;


        int LBAID {-1};
        int LCID {-1};

        float maxScaleDist, minScaleDist;

        bool isActive {true};

        bool inFrame {true};
        bool inFrameR {true};
        bool isOutlier {false};
        bool added {false};

        cv::Point2f predL, predR;
        float predAngleL, predAngleR;

        int scaleLevel {0};
        int prdScaleLevel {0};
        int scaleLevelL {0};
        int scaleLevelR {0};

        int keyFrameNb {0};
        const unsigned long kdx;
        const unsigned long idx;

        void update(KeyFrame* kF);
        void update(KeyFrame* kF, const bool back);
        int predictScale(float dist);
        void addConnection(KeyFrame* kF, const std::pair<int,int>& keyPos);
        void addConnectionB(KeyFrame* kF, const std::pair<int,int>& keyPos);

        void eraseKFConnection(KeyFrame* kF);
        void eraseKFConnectionB(KeyFrame* kF);
        void setActive(bool act);
        void SetInFrame(bool infr);
        void SetIsOutlier(bool isOut);
        bool getActive() const;
        bool GetIsOutlier() const;
        bool GetInFrame() const;
        void calcDescriptor();
        MapPoint(const Eigen::Vector4d& p, const cv::Mat& _desc, const cv::KeyPoint& obsK, const unsigned long _kdx, const unsigned long _idx);

        Eigen::Vector4d getWordPose4d() const;
        Eigen::Vector3d getWordPose3d() const;
        void updatePos(const Eigen::Vector3d& newPos, const Zed_Camera* zedPtr);
        void setWordPose4d(const Eigen::Vector4d& p);
        void setWordPose3d(const Eigen::Vector3d& p);
};

class Map
{
    private:

    public:

        bool aprilTagDetected {false};

        bool endOfFrames {false};

        std::unordered_map<unsigned long, KeyFrame*> keyFrames;
        std::unordered_map<unsigned long, MapPoint*> mapPoints;
        std::vector<MapPoint*> activeMapPoints;
        std::vector<MapPoint*> activeMapPointsB;
        std::vector<KeyFrame*> allFramesPoses;
        unsigned long kIdx {0};
        unsigned long pIdx {0};
        
        bool keyFrameAdded {false};
        bool keyFrameAddedMain {false};
        bool frameAdded {false};
        bool LBADone {false};
        int endLBAIdx {0};

        
        Eigen::Matrix4d LCPose = Eigen::Matrix4d::Identity();
        bool LCDone {false};
        bool LCStart {false};
        int LCCandIdx {-1};
        int endLCIdx {0};


        Map(){};
        void addMapPoint(MapPoint* mp);
        void addKeyFrame(KeyFrame* kF);
        void removeKeyFrame(int idx);
        mutable std::mutex mapMutex;

    protected:
};

} // namespace DC_VSLAM

#endif // MAP_H