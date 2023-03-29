#include "Map.h"

namespace DC_VSLAM
{

MapPoint::MapPoint(const Eigen::Vector4d& p, const cv::Mat& _desc, const cv::KeyPoint& obsK, const unsigned long _kdx, const unsigned long _idx) : wp(p), kdx(_kdx), idx(_idx)
{
    wp3d = Eigen::Vector3d(p(0), p(1), p(2));
    obs.push_back(obsK);
    desc.push_back(_desc);
}

int MapPoint::predictScale(float dist)
{
    float dif = maxScaleDist/dist;
    int scale = cvCeil(log(dif)/lastObsKF->logScale);
    if ( scale < 0 )
        scale = 0;
    else if ( scale >= lastObsKF->nScaleLev )
        scale = lastObsKF->nScaleLev - 1;
    // std::cout << "scale" << scale << std::endl;
    return scale;
}

void MapPoint::addConnection(KeyFrame* kF, const std::pair<int,int>& keyPos)
{
    if ( kFMatches.find(kF) == kFMatches.end() )
        kFMatches.insert(std::pair<KeyFrame*, std::pair<int,int>>(kF, keyPos));
    else
        kFMatches[kF] = keyPos;

    if ( keyPos.first >= 0 )
    {
        kF->localMapPoints[keyPos.first] = this;
        kF->unMatchedF[keyPos.first] = kdx;
    }
    if ( keyPos.second >= 0 )
    {
        kF->localMapPointsR[keyPos.second] = this;
        kF->unMatchedFR[keyPos.second] = kdx;
    }
}

void MapPoint::addConnectionB(KeyFrame* kF, const std::pair<int,int>& keyPos)
{
    if ( kFMatchesB.find(kF) == kFMatchesB.end() )
        kFMatchesB.insert(std::pair<KeyFrame*, std::pair<int,int>>(kF, keyPos));
    else
        kFMatchesB[kF] = keyPos;

    if ( keyPos.first >= 0 )
    {
        kF->localMapPointsB[keyPos.first] = this;
        kF->unMatchedFB[keyPos.first] = kdx;
    }
    if ( keyPos.second >= 0 )
    {
        kF->localMapPointsRB[keyPos.second] = this;
        kF->unMatchedFRB[keyPos.second] = kdx;
    }
}

void MapPoint::update(KeyFrame* kF)
{
    lastObsKF = kF;
    const TrackedKeys& keysLeft = kF->keys;
    Eigen::Vector3d pos = wp3d;
    pos = pos - kF->pose.pose.block<3,1>(0,3);
    const float dist = pos.norm();
    const std::pair<int, int>& idxs = kFMatches[kF];
    int level {0};
    lastObsKF = kF;
    if ( idxs.second >= 0 )
    {
        lastObsR = keysLeft.rightKeyPoints[idxs.second];
        scaleLevelR = keysLeft.rightKeyPoints[idxs.second].octave;
        level = scaleLevelR;
        if ( idxs.first < 0 )
        {
            lastObsL = lastObsR;
            scaleLevelL = scaleLevelR;
        }
    }
    if ( idxs.first >= 0 )
    {
        lastObsL = keysLeft.keyPoints[idxs.first];
        scaleLevelL = keysLeft.keyPoints[idxs.first].octave;
        level = scaleLevelL;
        if ( idxs.second < 0 )
        {
            lastObsR = lastObsL;
            scaleLevelR = scaleLevelL;
        }
    }

    const float scaleF = kF->scaleFactor[level];
    const int maxLevels = kF->nScaleLev;


    maxScaleDist = dist * scaleF;
    minScaleDist = maxScaleDist / kF->scaleFactor[maxLevels - 1];

    calcDescriptor();
}

void MapPoint::update(KeyFrame* kF, const bool back)
{
    lastObsKF = kF;
    const TrackedKeys& keysLeft = (back) ? kF->keysB : kF->keys;
    Eigen::Vector3d pos = wp3d;
    const Eigen::Vector3d camPose = (back) ? kF->backPose.block<3,1>(0,3) : kF->pose.pose.block<3,1>(0,3);
    pos = pos - camPose;
    const float dist = pos.norm();
    const std::pair<int, int>& idxs = (back) ? kFMatchesB[kF] : kFMatches[kF];
    int level {0};
    lastObsKF = kF;
    if ( idxs.second >= 0 )
    {
        lastObsR = keysLeft.rightKeyPoints[idxs.second];
        scaleLevelR = keysLeft.rightKeyPoints[idxs.second].octave;
        level = scaleLevelR;
        if ( idxs.first < 0 )
        {
            lastObsL = lastObsR;
            scaleLevelL = scaleLevelR;
        }
    }
    if ( idxs.first >= 0 )
    {
        lastObsL = keysLeft.keyPoints[idxs.first];
        scaleLevelL = keysLeft.keyPoints[idxs.first].octave;
        level = scaleLevelL;
        if ( idxs.second < 0 )
        {
            lastObsR = lastObsL;
            scaleLevelR = scaleLevelL;
        }
    }

    const float scaleF = kF->scaleFactor[level];
    const int maxLevels = kF->nScaleLev;


    maxScaleDist = dist * scaleF;
    minScaleDist = maxScaleDist / kF->scaleFactor[maxLevels - 1];

    calcDescriptor();
}

void MapPoint::calcDescriptor()
{
        // Retrieve all observed descriptors
    std::vector<cv::Mat> vDescriptors;

    std::unordered_map<KeyFrame*,std::pair<int,int>> observations = kFMatches;
    std::unordered_map<KeyFrame*,std::pair<int,int>> observationsB = kFMatchesB;

    if(observations.empty())
        return;

    vDescriptors.reserve(observations.size() + observationsB.size());

    for(std::unordered_map<KeyFrame*,std::pair<int,int>> ::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        std::pair<int,int> indexes = mit -> second;
        int leftIndex = indexes.first, rightIndex = indexes.second;

        if(leftIndex != -1){
            vDescriptors.push_back(pKF->keys.Desc.row(leftIndex));
        }
        if(rightIndex != -1){
            vDescriptors.push_back(pKF->keys.rightDesc.row(rightIndex));
        }
    }

    for(std::unordered_map<KeyFrame*,std::pair<int,int>> ::iterator mit=observationsB.begin(), mend=observationsB.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        std::pair<int,int> indexes = mit -> second;
        int leftIndex = indexes.first, rightIndex = indexes.second;

        if(leftIndex != -1){
            vDescriptors.push_back(pKF->keysB.Desc.row(leftIndex));
        }
        if(rightIndex != -1){
            vDescriptors.push_back(pKF->keysB.rightDesc.row(rightIndex));
        }
    }


    if(vDescriptors.empty())
        return;

    // Compute distances between them
    const size_t N = vDescriptors.size();

    float Distances[N][N];
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
            int distij = FeatureMatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {
        std::vector<int> vDists(Distances[i],Distances[i]+N);
        std::sort(vDists.begin(),vDists.end());
        int median = vDists[0.5*(N-1)];

        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        desc = vDescriptors[BestIdx].clone();
    }
}

void MapPoint::updatePos(const Eigen::Vector3d& newPos, const Zed_Camera* zedPtr)
{
    setWordPose3d(newPos);
    std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator it;
    std::unordered_map<KeyFrame*, std::pair<int,int>>::const_iterator end(kFMatches.end());
    for ( it = kFMatches.begin(); it != end; it++)
    {
        KeyFrame* kFcand = it->first;
        const std::pair<int,int> keyPos = it->second;
        TrackedKeys& tKeys = kFcand->keys;
        if ( tKeys.estimatedDepth[keyPos.first] <= 0 )
            continue;
        Eigen::Vector4d pCam = kFcand->pose.getInvPose() * wp;
        tKeys.estimatedDepth[keyPos.first] = pCam(2);
        if ( pCam(2) <= zedPtr->mBaseline * 40)
        {
            tKeys.close[keyPos.first] = true;
        }
    }

    std::unordered_map<KeyFrame*, std::pair<int,int>>::const_iterator endB(kFMatchesB.end());
    for ( it = kFMatchesB.begin(); it != endB; it++)
    {
        KeyFrame* kFcand = it->first;
        const std::pair<int,int> keyPos = it->second;
        TrackedKeys& tKeys = kFcand->keysB;
        if ( tKeys.estimatedDepth[keyPos.first] <= 0 )
            continue;
        Eigen::Vector4d pCam = kFcand->backPoseInv * wp;
        tKeys.estimatedDepth[keyPos.first] = pCam(2);
        if ( pCam(2) <= zedPtr->mBaseline * 40)
        {
            tKeys.close[keyPos.first] = true;
        }
    }

    calcDescriptor();

}

void MapPoint::eraseKFConnection(KeyFrame* kF)
{
    kFMatches.erase(kF);
}

void MapPoint::eraseKFConnectionB(KeyFrame* kF)
{
    kFMatchesB.erase(kF);
}

bool MapPoint::GetInFrame() const
{
    return inFrame;
}

bool MapPoint::GetIsOutlier() const
{
    return isOutlier;
}

bool MapPoint::getActive() const
{
    return isActive;
}

void MapPoint::SetInFrame(bool infr)
{
    inFrame = infr;
}

void MapPoint::SetIsOutlier(bool isOut)
{
    isOutlier = isOut;
}

void MapPoint::setActive(bool act)
{
    isActive = act;
}

Eigen::Vector4d MapPoint::getWordPose4d() const
{
    return wp;
}

Eigen::Vector3d MapPoint::getWordPose3d() const
{
    return wp3d;
}



void MapPoint::setWordPose4d(const Eigen::Vector4d& p)
{
    wp = p;
    wp3d = Eigen::Vector3d(p(0), p(1), p(2));
}

void MapPoint::setWordPose3d(const Eigen::Vector3d& p)
{
    wp3d = p;
    wp = Eigen::Vector4d(p(0), p(1), p(2), 1.0);
}

void Map::addMapPoint(MapPoint* mp)
{
    mp->added = true;
    mapPoints.insert(std::pair<unsigned long, MapPoint*>(pIdx, mp));
    pIdx++;
}

void Map::removeKeyFrame(int idx)
{
    keyFrames.erase((unsigned long) idx);
}

void Map::addKeyFrame(KeyFrame* kF)
{
    keyFrames.insert(std::pair<unsigned long, KeyFrame*>(kIdx, kF));
    kIdx ++;
}

} // namespace DC_VSLAM