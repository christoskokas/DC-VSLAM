#include "System.h"


namespace DC_VSLAM
{

System::System(ConfigFile* _mConf, bool _LC) : LC(_LC)
{
    mConf = _mConf;

    int nFeatures = mConf->getValue<int>("FE", "nFeatures");
    int nLevels = mConf->getValue<int>("FE", "nLevels");
    float imScale = mConf->getValue<float>("FE", "imScale");
    int edgeThreshold = mConf->getValue<int>("FE", "edgeThreshold");
    int maxFastThreshold = mConf->getValue<int>("FE", "maxFastThreshold");
    int minFastThreshold = mConf->getValue<int>("FE", "minFastThreshold");
    int patchSize = mConf->getValue<int>("FE", "patchSize");

    mZedCamera = new Zed_Camera(mConf);

    mFrame = new ViewFrame;

    map = new Map();

    feLeft = new FeatureExtractor(nFeatures, nLevels, imScale, edgeThreshold, patchSize, maxFastThreshold, minFastThreshold);
    feRight = new FeatureExtractor(nFeatures, nLevels, imScale, edgeThreshold, patchSize, maxFastThreshold, minFastThreshold);

    featTracker = new FeatureTracker(mZedCamera, feLeft, feRight, map);


    fm = new FeatureMatcher(mZedCamera, feLeft, feRight, mZedCamera->mHeight);

    localMap = new LocalMapper(map, mZedCamera, fm);
    loopCl = new LocalMapper(map, mZedCamera, fm);

    Visual = new std::thread(&DC_VSLAM::ViewFrame::pangoQuit, mFrame, mZedCamera, map);

    LocalMapping = new std::thread(&DC_VSLAM::LocalMapper::beginLocalMapping, localMap);
    if ( LC )
        LoopClosure = new std::thread(&DC_VSLAM::LocalMapper::beginLoopClosure, loopCl);
}

System::System(ConfigFile* _mConf, bool _LC, bool multi) : LC(_LC)
{
    mConf = _mConf;

    int nFeatures = mConf->getValue<int>("FE", "nFeatures");
    int nLevels = mConf->getValue<int>("FE", "nLevels");
    float imScale = mConf->getValue<float>("FE", "imScale");
    int edgeThreshold = mConf->getValue<int>("FE", "edgeThreshold");
    int maxFastThreshold = mConf->getValue<int>("FE", "maxFastThreshold");
    int minFastThreshold = mConf->getValue<int>("FE", "minFastThreshold");
    int patchSize = mConf->getValue<int>("FE", "patchSize");

    mZedCamera = new Zed_Camera(mConf, false);
    mZedCameraB = new Zed_Camera(mConf, true);

    mFrame = new ViewFrame;

    map = new Map();

    feLeft = new FeatureExtractor(nFeatures, nLevels, imScale, edgeThreshold, patchSize, maxFastThreshold, minFastThreshold);
    feRight = new FeatureExtractor(nFeatures, nLevels, imScale, edgeThreshold, patchSize, maxFastThreshold, minFastThreshold);

    feLeftB = new FeatureExtractor(nFeatures, nLevels, imScale, edgeThreshold, patchSize, maxFastThreshold, minFastThreshold);
    feRightB = new FeatureExtractor(nFeatures, nLevels, imScale, edgeThreshold, patchSize, maxFastThreshold, minFastThreshold);

    featTracker = new FeatureTracker(mZedCamera, mZedCameraB, feLeft, feRight, feLeftB, feRightB, map);


    fm = new FeatureMatcher(mZedCamera, feLeft, feRight, mZedCamera->mHeight);

    localMap = new LocalMapper(map, mZedCamera, mZedCameraB, fm);
    loopCl = new LocalMapper(map, mZedCamera, mZedCameraB, fm);

    Visual = new std::thread(&DC_VSLAM::ViewFrame::pangoQuitMulti, mFrame, mZedCamera, mZedCameraB, map);

    LocalMapping = new std::thread(&DC_VSLAM::LocalMapper::beginLocalMappingB, localMap);

    if ( LC )
        LoopClosure = new std::thread(&DC_VSLAM::LocalMapper::beginLoopClosureB, loopCl);
    
}

void System::saveTrajectory(const std::string& filepath)
{
    std::vector<KeyFrame*>& allFrames = map->allFramesPoses;
    KeyFrame* closeKF = allFrames[0];
    std::ofstream datafile(filepath);
    std::vector<KeyFrame*>::iterator it;
    std::vector<KeyFrame*>::const_iterator end(allFrames.end());
    for ( it = allFrames.begin(); it != end; it ++)
    {
        KeyFrame* candKF = *it;
        Eigen::Matrix4d matT;
        if ( candKF->keyF )
        {
            matT = candKF->pose.pose;
            closeKF = candKF;
        }
        else
        {
            matT = (closeKF->pose.getPose() * candKF->pose.refPose);
        }
        Eigen::Matrix4d mat = matT.transpose();
        for (int32_t i{0}; i < 12; i ++)
        {
            if ( i == 0 )
                datafile << mat(i);
            else
                datafile << " " << mat(i);
        }
        datafile << '\n';
    }
    datafile.close();

}

void System::saveTrajectoryAndPosition(const std::string& filepath, const std::string& filepathPosition)
{
    std::vector<KeyFrame*>& allFrames = map->allFramesPoses;
    KeyFrame* closeKF = allFrames[0];
    std::ofstream datafile(filepath);
    std::ofstream datafilePos(filepathPosition);
    std::vector<KeyFrame*>::iterator it;
    std::vector<KeyFrame*>::const_iterator end(allFrames.end());
    for ( it = allFrames.begin(); it != end; it ++)
    {
        KeyFrame* candKF = *it;
        Eigen::Matrix4d matT;
        if ( candKF->keyF )
        {
            matT = candKF->pose.pose;
            closeKF = candKF;
        }
        else
        {
            matT = (closeKF->pose.getPose() * candKF->pose.refPose);
        }
        Eigen::Matrix4d mat = matT.transpose();
        for (int32_t i{0}; i < 12; i ++)
        {
            if ( i == 0 )
                datafile << mat(i);
            else
                datafile << " " << mat(i);
            if ( i == 3 || i == 7 || i == 11 )
                datafilePos << mat(i) << " ";
        }
        datafile << '\n';
        datafilePos << '\n';
    }
    datafile.close();
    datafilePos.close();

}

void System::trackNewImage(const cv::Mat& imLRect, const cv::Mat& imRRect, const int frameNumb)
{
    featTracker->TrackImageT(imLRect, imRRect, frameNumb);
}

void System::trackNewImageMutli(const cv::Mat& imLRect, const cv::Mat& imRRect, const cv::Mat& imLRectB, const cv::Mat& imRRectB, const int frameNumb)
{
    featTracker->TrackImageTB(imLRect, imRRect, imLRectB, imRRectB, frameNumb);
}

void System::exitSystem()
{
    mFrame->stopRequested = true;
    localMap->stopRequested = true;
    if ( LC )
        loopCl->stopRequested = true;
    Visual->join();
    LocalMapping->join();
    if ( LC )
        LoopClosure->join();
}

} // namespace DC_VSLAM