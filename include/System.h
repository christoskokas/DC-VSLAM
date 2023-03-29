#ifndef SYSTEM_H
#define SYSTEM_H

#include "Settings.h"
#include "Camera.h"
#include "Frame.h"
#include "FeatureTracker.h"
#include "Map.h"
#include "LocalBA.h"
#include <thread>
#include <string>



namespace DC_VSLAM
{

class System
{

        private:

        public:

        System(ConfigFile* _mConf, bool _LC);
        System(ConfigFile* _mConf, bool _LC, bool multi);

        void trackNewImage(const cv::Mat& imLRect, const cv::Mat& imRRect, const int frameNumb);
        void trackNewImageMutli(const cv::Mat& imLRect, const cv::Mat& imRRect, const cv::Mat& imLRectB, const cv::Mat& imRRectB, const int frameNumb);
        void saveTrajectory(const std::string& filepath);
        void saveTrajectoryAndPosition(const std::string& filepath, const std::string& filepathPosition);

        void exitSystem();

        // void ImagesCallback(const sensor_msgs::ImageConstPtr& lIm, const sensor_msgs::ImageConstPtr& rIm);
        bool LC {false};
        std::thread* Visual;
        std::thread* Tracking;
        std::thread* LocalMapping;
        std::thread* LoopClosure;

        std::thread* FeatTrack;
        std::thread* FeatTrackB;

        FeatureTracker* featTracker;
        FeatureTracker* featTrackerB;

        ViewFrame* mFrame;

        Zed_Camera* mZedCamera;
        Zed_Camera* mZedCameraB;

        ConfigFile* mConf;

        Map* map;

        LocalMapper* localMap;
        LocalMapper* loopCl;

        FeatureExtractor* feLeft;
        FeatureExtractor* feLeftB;

        FeatureExtractor* feRight;
        FeatureExtractor* feRightB;
        
        FeatureMatcher* fm;

};

} // namespace DC_VSLAM



#endif // SYSTEM_H