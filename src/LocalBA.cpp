#include "LocalBA.h"

namespace DC_VSLAM
{

LocalMapper::LocalMapper(Map* _map, Zed_Camera* _zedPtr, FeatureMatcher* _fm) : map(_map), zedPtr(_zedPtr), fm(_fm), fx(_zedPtr->cameraLeft.fx), fy(_zedPtr->cameraLeft.fy), cx(_zedPtr->cameraLeft.cx), cy(_zedPtr->cameraLeft.cy)
{

}

LocalMapper::LocalMapper(Map* _map, Zed_Camera* _zedPtr, Zed_Camera* _zedPtrB, FeatureMatcher* _fm) : map(_map), zedPtr(_zedPtr), zedPtrB(_zedPtrB), fm(_fm), fx(_zedPtr->cameraLeft.fx), fy(_zedPtr->cameraLeft.fy), cx(_zedPtr->cameraLeft.cx), cy(_zedPtr->cameraLeft.cy)
{

}

void LocalMapper::calcProjMatricesR(std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>>& projMatrices, std::vector<KeyFrame*>& actKeyF)
{

    Eigen::Matrix<double,3,3>& K = zedPtr->cameraLeft.intrinsics;
    Eigen::Matrix4d projL = Eigen::Matrix4d::Identity();
    projL.block<3,3>(0,0) = K;
    Eigen::Matrix4d projR = Eigen::Matrix4d::Identity();
    projR.block<3,3>(0,0) = K;
    std::vector<KeyFrame*>::const_iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        Eigen::Matrix<double,4,4> extr2 = (*it)->pose.poseInverse;
        extr2 = projL * extr2;
        Eigen::Matrix<double,3,4> extr = extr2.block<3,4>(0,0);
        Eigen::Matrix<double,4,4> extrRight = ((*it)->pose.pose * zedPtr->extrinsics).inverse();
        extrRight =  projR * extrRight;
        Eigen::Matrix<double,3,4> extrR = extrRight.block<3,4>(0,0);
        projMatrices.emplace(*it, std::make_pair(extr, extrR));
    }
}

void LocalMapper::calcProjMatricesRB(const Zed_Camera* zedCam,std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>>& projMatrices, std::vector<KeyFrame*>& actKeyF,const bool back)
{
    const Eigen::Matrix<double,3,3>& K = zedCam->cameraLeft.intrinsics;
    Eigen::Matrix4d projL = Eigen::Matrix4d::Identity();
    projL.block<3,3>(0,0) = K;
    Eigen::Matrix4d projR = Eigen::Matrix4d::Identity();
    projR.block<3,3>(0,0) = K;
    std::vector<KeyFrame*>::const_iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        const Eigen::Matrix4d& poseCam = (back) ? (*it)->backPose : (*it)->pose.pose;
        const Eigen::Matrix4d& poseCamInv = (back) ? (*it)->backPoseInv : (*it)->pose.poseInverse;
        Eigen::Matrix<double,4,4> extr2 = poseCamInv;
        extr2 = projL * extr2;
        Eigen::Matrix<double,3,4> extr = extr2.block<3,4>(0,0);
        Eigen::Matrix<double,4,4> extrRight = (poseCam * zedCam->extrinsics).inverse();
        extrRight =  projR * extrRight;
        Eigen::Matrix<double,3,4> extrR = extrRight.block<3,4>(0,0);
        projMatrices.emplace(*it, std::make_pair(extr, extrR));
    }
}

void LocalMapper::processMatchesR(std::vector<std::pair<DC_VSLAM::KeyFrame *, std::pair<int, int>>>& matchesOfPoint, std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>>& allProjMatrices, std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& points)
{
    proj_matrices.reserve(matchesOfPoint.size());
    points.reserve(matchesOfPoint.size());
    std::vector<std::pair<DC_VSLAM::KeyFrame *, std::pair<int, int>>>::const_iterator it, end(matchesOfPoint.end());
    for ( it = matchesOfPoint.begin(); it != end; it++)
    {
        KeyFrame* kF = it->first;
        const TrackedKeys& keys = kF->keys;
        const std::pair<int,int>& keyPos = it->second;
        if ( keyPos.first >= 0 )
        {
            Eigen::Vector2d vec2d((double)keys.keyPoints[keyPos.first].pt.x, (double)keys.keyPoints[keyPos.first].pt.y);
            points.emplace_back(vec2d);
            proj_matrices.emplace_back(allProjMatrices.at(kF).first);
        }

        if ( keyPos.second >= 0 )
        {
            Eigen::Vector2d vec2d((double)keys.rightKeyPoints[keyPos.second].pt.x, (double)keys.rightKeyPoints[keyPos.second].pt.y);
            points.emplace_back(vec2d);
            proj_matrices.emplace_back(allProjMatrices.at(kF).second);
        }
    }
}

void LocalMapper::processMatchesRB(std::vector<std::pair<DC_VSLAM::KeyFrame *, std::pair<int, int>>>& matchesOfPoint, std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>>& allProjMatrices, std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& points, const bool back)
{
    proj_matrices.reserve(matchesOfPoint.size());
    points.reserve(matchesOfPoint.size());
    std::vector<std::pair<DC_VSLAM::KeyFrame *, std::pair<int, int>>>::const_iterator it, end(matchesOfPoint.end());
    for ( it = matchesOfPoint.begin(); it != end; it++)
    {
        KeyFrame* kF = it->first;
        const TrackedKeys& keys = (back) ? kF->keysB : kF->keys;
        const std::pair<int,int>& keyPos = it->second;
        if ( keyPos.first >= 0 )
        {
            Eigen::Vector2d vec2d((double)keys.keyPoints[keyPos.first].pt.x, (double)keys.keyPoints[keyPos.first].pt.y);
            points.emplace_back(vec2d);
            proj_matrices.emplace_back(allProjMatrices.at(kF).first);
        }

        if ( keyPos.second >= 0 )
        {
            Eigen::Vector2d vec2d((double)keys.rightKeyPoints[keyPos.second].pt.x, (double)keys.rightKeyPoints[keyPos.second].pt.y);
            points.emplace_back(vec2d);
            proj_matrices.emplace_back(allProjMatrices.at(kF).second);
        }
    }
}

bool LocalMapper::checkReprojErrNewR(KeyFrame* lastKF, Eigen::Vector4d& calcVec, std::vector<std::pair<KeyFrame *, std::pair<int, int>>>& matchesOfPoint, const std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& pointsVec)
{
    int count {0};
    bool correctKF {false};
    const unsigned long lastKFNumb = lastKF->numb;
    int projCount {0};
    for (size_t i {0}, end{matchesOfPoint.size()}; i < end; i++)
    {
        std::pair<KeyFrame *, std::pair<int, int>>& match = matchesOfPoint[i];
        KeyFrame* kFCand = matchesOfPoint[i].first;
        const TrackedKeys& keys = kFCand->keys;
        
        std::pair<int,int>& keyPos = matchesOfPoint[i].second;
        const unsigned long kFCandNumb {kFCand->numb};
        bool cor {false};
        if ( keyPos.first >= 0 )
        {
            Eigen::Vector3d p3dnew = proj_matrices[projCount] * calcVec;
            p3dnew = p3dnew/p3dnew(2);
            double err1,err2;
            err1 = pointsVec[projCount](0) - p3dnew(0);
            err2 = pointsVec[projCount](1) - p3dnew(1);
            const int oct = keys.keyPoints[keyPos.first].octave;
            const double weight = (double)kFCand->sigmaFactor[oct];
            float err = err1*err1 + err2*err2;
            projCount ++;

            if ( err > reprjThreshold * weight )
            {
                keyPos.first = -1;
            }
            else
            {
                matchesOfPoint[count] = match;
                cor = true;
                if ( kFCandNumb == lastKFNumb )
                    correctKF = true;
            }
        }
        if ( keyPos.second >= 0 )
        {
            Eigen::Vector3d p3dnew = proj_matrices[projCount] * calcVec;
            p3dnew = p3dnew/p3dnew(2);
            double err1,err2;
            err1 = pointsVec[projCount](0) - p3dnew(0);
            err2 = pointsVec[projCount](1) - p3dnew(1);
            const int oct = keys.rightKeyPoints[keyPos.second].octave;
            const double weight = lastKF->sigmaFactor[oct];
            float err = err1*err1 + err2*err2;
            projCount ++;

            if ( err > reprjThreshold * weight )
            {
                keyPos.second = -1;
            }
            else
            {
                matchesOfPoint[count] = match;
                cor = true;
                if ( kFCandNumb == lastKFNumb )
                    correctKF = true;
            }
        }
        if ( cor )
            count++;
    }
    matchesOfPoint.resize(count);
    if ( count >= minCount  && correctKF )
        return true;
    else
        return false;
}

bool LocalMapper::checkReprojErrNewRB(KeyFrame* lastKF, Eigen::Vector4d& calcVec, std::vector<std::pair<KeyFrame *, std::pair<int, int>>>& matchesOfPoint, const std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& pointsVec, const bool back)
{
    int count {0};
    bool correctKF {false};
    const unsigned long lastKFNumb = lastKF->numb;
    int projCount {0};
    for (size_t i {0}, end{matchesOfPoint.size()}; i < end; i++)
    {
        std::pair<KeyFrame *, std::pair<int, int>>& match = matchesOfPoint[i];
        KeyFrame* kFCand = matchesOfPoint[i].first;
        const TrackedKeys& keys = (back) ? kFCand->keysB : kFCand->keys;
        
        std::pair<int,int>& keyPos = matchesOfPoint[i].second;
        const unsigned long kFCandNumb {kFCand->numb};
        bool cor {false};
        if ( keyPos.first >= 0 )
        {
            Eigen::Vector3d p3dnew = proj_matrices[projCount] * calcVec;
            p3dnew = p3dnew/p3dnew(2);
            double err1,err2;
            err1 = pointsVec[projCount](0) - p3dnew(0);
            err2 = pointsVec[projCount](1) - p3dnew(1);
            const int oct = keys.keyPoints[keyPos.first].octave;
            const double weight = (double)kFCand->sigmaFactor[oct];
            float err = err1*err1 + err2*err2;
            projCount ++;

            if ( err > reprjThreshold * weight )
            {
                keyPos.first = -1;
            }
            else
            {
                matchesOfPoint[count] = match;
                cor = true;
                if ( kFCandNumb == lastKFNumb )
                    correctKF = true;
            }
        }
        if ( keyPos.second >= 0 )
        {
            Eigen::Vector3d p3dnew = proj_matrices[projCount] * calcVec;
            p3dnew = p3dnew/p3dnew(2);
            double err1,err2;
            err1 = pointsVec[projCount](0) - p3dnew(0);
            err2 = pointsVec[projCount](1) - p3dnew(1);
            const int oct = keys.rightKeyPoints[keyPos.second].octave;
            const double weight = lastKF->sigmaFactor[oct];
            float err = err1*err1 + err2*err2;
            projCount ++;

            if ( err > reprjThreshold * weight )
            {
                keyPos.second = -1;
            }
            else
            {
                matchesOfPoint[count] = match;
                cor = true;
                if ( kFCandNumb == lastKFNumb )
                    correctKF = true;
            }
        }
        if ( cor )
            count++;
    }
    matchesOfPoint.resize(count);
    if ( count >= minCount  && correctKF )
        return true;
    else
        return false;
}

void LocalMapper::addMultiViewMapPointsR(const Eigen::Vector4d& posW, const std::vector<std::pair<DC_VSLAM::KeyFrame *, std::pair<int, int>>>& matchesOfPoint, std::vector<MapPoint*>& pointsToAdd, KeyFrame* lastKF, const size_t& mpPos)
{
    const TrackedKeys& temp = lastKF->keys; 
    static unsigned long mpIdx {map->pIdx};
    const unsigned long lastKFNumb {lastKF->numb};
    MapPoint* mp = nullptr;
    for (size_t i {0}, end{matchesOfPoint.size()}; i < end; i++)
    {
        KeyFrame* kFCand = matchesOfPoint[i].first;
        const std::pair<int,int>& keyPos = matchesOfPoint[i].second;
        if ( kFCand->numb == lastKFNumb )
        {
            if ( keyPos.first >= 0 )
            {
                mp = new MapPoint(posW, temp.Desc.row(keyPos.first),temp.keyPoints[keyPos.first], lastKF->numb, mpIdx);
            }
            else if ( keyPos.second >= 0 )
            {
                mp = new MapPoint(posW, temp.rightDesc.row(keyPos.second),temp.rightKeyPoints[keyPos.second], lastKF->numb, mpIdx);
            }
            break;
        }
    }
    if ( !mp )
        return;

    mpIdx++;
    for (size_t i {0}, end{matchesOfPoint.size()}; i < end; i++)
    {
        KeyFrame* kFCand = matchesOfPoint[i].first;
        const std::pair<int,int>& keyPos = matchesOfPoint[i].second;
        mp->kFMatches.insert(std::pair<KeyFrame*, std::pair<int,int>>(kFCand, keyPos));
    }
    mp->update(lastKF);
    pointsToAdd[mpPos] = mp;
}

void LocalMapper::addMultiViewMapPointsRB(const Eigen::Vector4d& posW, const std::vector<std::pair<DC_VSLAM::KeyFrame *, std::pair<int, int>>>& matchesOfPoint, std::vector<MapPoint*>& pointsToAdd, KeyFrame* lastKF, const size_t& mpPos, const bool back)
{
    const TrackedKeys& temp = (back) ? lastKF->keysB : lastKF->keys; 
    static unsigned long mpIdx {map->pIdx};
    const unsigned long lastKFNumb {lastKF->numb};
    MapPoint* mp = nullptr;
    for (size_t i {0}, end{matchesOfPoint.size()}; i < end; i++)
    {
        KeyFrame* kFCand = matchesOfPoint[i].first;
        const std::pair<int,int>& keyPos = matchesOfPoint[i].second;
        if ( kFCand->numb == lastKFNumb )
        {
            if ( keyPos.first >= 0 )
            {
                mp = new MapPoint(posW, temp.Desc.row(keyPos.first),temp.keyPoints[keyPos.first], lastKF->numb, mpIdx);
            }
            else if ( keyPos.second >= 0 )
            {
                mp = new MapPoint(posW, temp.rightDesc.row(keyPos.second),temp.rightKeyPoints[keyPos.second], lastKF->numb, mpIdx);
            }
            break;
        }
    }
    if ( !mp )
        return;

    mpIdx++;
    for (size_t i {0}, end{matchesOfPoint.size()}; i < end; i++)
    {
        KeyFrame* kFCand = matchesOfPoint[i].first;
        const std::pair<int,int>& keyPos = matchesOfPoint[i].second;
        if ( back )
            mp->kFMatchesB.insert(std::pair<KeyFrame*, std::pair<int,int>>(kFCand, keyPos));
        else
            mp->kFMatches.insert(std::pair<KeyFrame*, std::pair<int,int>>(kFCand, keyPos));

    }
    mp->update(lastKF, back);
    pointsToAdd[mpPos] = mp;
}

void LocalMapper::triangulateCeresNew(Eigen::Vector3d& p3d, const std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, const std::vector<Eigen::Vector2d>& obs, const Eigen::Matrix4d& lastKFPose, bool first)
{
    const Eigen::Matrix4d& camPose = lastKFPose;
    ceres::Problem problem;
    ceres::LossFunction* loss_function = nullptr;
    if ( first )
        loss_function = new ceres::HuberLoss(sqrt(7.815f));
    for (size_t i {0}, end{obs.size()}; i < end; i ++)
    {
        ceres::CostFunction* costf = MultiViewTriang::Create(camPose, proj_matrices[i], obs[i]);
        problem.AddResidualBlock(costf, loss_function /* squared loss */,p3d.data());

    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 20;
    problem.SetParameterLowerBound(p3d.data(), 2, 0.1);
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Eigen::Vector4d p4d(p3d(0), p3d(1), p3d(2), 1.0);
    p4d = lastKFPose * p4d;
    p3d(0) = p4d(0);
    p3d(1) = p4d(1);
    p3d(2) = p4d(2);

}

void LocalMapper::addNewMapPoints(KeyFrame* lastKF, std::vector<MapPoint*>& pointsToAdd, std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs)
{
    int newMapPointsCount {0};
    std::lock_guard<std::mutex> lock(map->mapMutex);
    for (size_t i{0}, end{pointsToAdd.size()}; i < end; i++ )
    {
        MapPoint* newMp = pointsToAdd[i];
        if (!newMp )
            continue;
        std::unordered_map<DC_VSLAM::KeyFrame *, std::pair<int, int>>::iterator it, endMp(newMp->kFMatches.end());
        for ( it = newMp->kFMatches.begin(); it != endMp; it++)
        {
            KeyFrame* kFCand = it->first;
            std::pair<int,int>& keyPos = it->second;
            newMp->addConnection(kFCand, keyPos);
        }
        map->activeMapPoints.emplace_back(newMp);
        map->addMapPoint(newMp);
        newMapPointsCount ++;
    }

}

void LocalMapper::addNewMapPointsB(KeyFrame* lastKF, std::vector<MapPoint*>& pointsToAdd, std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs, const bool back)
{
    std::lock_guard<std::mutex> lock(map->mapMutex);
    for (size_t i{0}, end{pointsToAdd.size()}; i < end; i++ )
    {
        MapPoint* newMp = pointsToAdd[i];
        if (!newMp )
            continue;
        std::unordered_map<DC_VSLAM::KeyFrame *, std::pair<int, int>>::iterator it = (back) ? newMp->kFMatchesB.begin() : newMp->kFMatches.begin();
        std::unordered_map<DC_VSLAM::KeyFrame *, std::pair<int, int>>::iterator endMp = (back) ? newMp->kFMatchesB.end() : newMp->kFMatches.end();
        for (; it != endMp; it++)
        {
            KeyFrame* kFCand = it->first;
            std::pair<int,int>& keyPos = it->second;
            if ( back )
                newMp->addConnectionB(kFCand, keyPos);
            else
                newMp->addConnection(kFCand, keyPos);

        }
        if ( back )
            map->activeMapPointsB.emplace_back(newMp);
        else
            map->activeMapPoints.emplace_back(newMp);
        map->addMapPoint(newMp);
    }
}

void LocalMapper::calcAllMpsOfKFROnlyEst(std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs, KeyFrame* lastKF, const int kFsize, std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<float>& maxDistsScale)
{
    const size_t keysSize {lastKF->keys.keyPoints.size()};
    const size_t RkeysSize {lastKF->keys.rightKeyPoints.size()};
    const TrackedKeys& keys = lastKF->keys;
    p4d.reserve(keysSize + RkeysSize);
    maxDistsScale.reserve(keysSize + RkeysSize);
    for ( size_t i{0}; i < keysSize; i++)
    {
        MapPoint* mp = lastKF->localMapPoints[i];
        if ( !mp )
        {
            double zp;
            int rIdx {-1};
            if ( keys.estimatedDepth[i] > 0 )
            {
                rIdx = keys.rightIdxs[i];
                zp = (double)keys.estimatedDepth[i];
            }
            else
                continue;
            const double xp = (double)(((double)keys.keyPoints[i].pt.x-cx)*zp/fx);
            const double yp = (double)(((double)keys.keyPoints[i].pt.y-cy)*zp/fy);
            Eigen::Vector4d p4dcam(xp, yp, zp, 1);
            p4dcam = lastKF->pose.pose * p4dcam;
            p4d.emplace_back(p4dcam, std::make_pair((int)i, rIdx));
            Eigen::Vector3d pos = p4dcam.block<3,1>(0,0);
            pos = pos - lastKF->pose.pose.block<3,1>(0,3);
            float dist = pos.norm();
            int level = keys.keyPoints[i].octave;
            dist *= lastKF->scaleFactor[level];
            maxDistsScale.emplace_back(dist);
            continue;
        }
        if ( lastKF->unMatchedF[i] >= 0 )
            continue;
        const int rIdx {keys.rightIdxs[i]};
        p4d.emplace_back(mp->getWordPose4d(), std::make_pair((int)i, rIdx));
        Eigen::Vector3d pos = mp->getWordPose4d().block<3,1>(0,0);
        pos = pos - lastKF->pose.pose.block<3,1>(0,3);
        float dist = pos.norm();
        int level = keys.keyPoints[i].octave;
        dist *= lastKF->scaleFactor[level];
        maxDistsScale.emplace_back(dist);
    }
    const size_t allp4dsize {p4d.size()};
    matchedIdxs = std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>(allp4dsize,std::vector<std::pair<KeyFrame*,std::pair<int, int>>>());
    for ( size_t i {0}; i < allp4dsize; i++)
    {
        matchedIdxs[i].reserve(10);
        std::pair<int,int> keyPos = p4d[i].second;
        matchedIdxs[i].emplace_back(lastKF,keyPos);
    }
}

void LocalMapper::calcAllMpsOfKFROnlyEstB(const Zed_Camera* zedCam, std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs, KeyFrame* lastKF, const int kFsize, std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<float>& maxDistsScale, const bool back)
{
    const Eigen::Matrix4d& pose4d = (back) ? lastKF->backPose : lastKF->pose.pose;
    const std::vector<int>& unMatchVec = (back) ? lastKF->unMatchedFB : lastKF->unMatchedF;
    const std::vector<MapPoint*>& localMPs = (back) ? lastKF->localMapPointsB : lastKF->localMapPoints;
    const TrackedKeys& keys = (back) ? lastKF->keysB : lastKF->keys;
    const Eigen::Vector3d traToRem = pose4d.block<3,1>(0,3);
    const size_t keysSize {keys.keyPoints.size()};
    const size_t RkeysSize {keys.rightKeyPoints.size()};
    p4d.reserve(keysSize + RkeysSize);
    maxDistsScale.reserve(keysSize + RkeysSize);
    for ( size_t i{0}; i < keysSize; i++)
    {
        MapPoint* mp = localMPs[i];
        if ( !mp )
        {
            double zp;
            int rIdx {-1};
            if ( keys.estimatedDepth[i] > 0 )
            {
                rIdx = keys.rightIdxs[i];
                zp = (double)keys.estimatedDepth[i];
            }
            else
                continue;
            const double xp = (double)(((double)keys.keyPoints[i].pt.x-cx)*zp/fx);
            const double yp = (double)(((double)keys.keyPoints[i].pt.y-cy)*zp/fy);
            Eigen::Vector4d p4dcam(xp, yp, zp, 1);
            p4dcam = pose4d * p4dcam;
            p4d.emplace_back(p4dcam, std::make_pair((int)i, rIdx));
            Eigen::Vector3d pos = p4dcam.block<3,1>(0,0);
            pos = pos - traToRem;
            float dist = pos.norm();
            int level = keys.keyPoints[i].octave;
            dist *= lastKF->scaleFactor[level];
            maxDistsScale.emplace_back(dist);
            continue;
        }
        if ( unMatchVec[i] >= 0 )
            continue;
        const int rIdx {keys.rightIdxs[i]};
        p4d.emplace_back(mp->getWordPose4d(), std::make_pair((int)i, rIdx));
        Eigen::Vector3d pos = mp->getWordPose4d().block<3,1>(0,0);
        pos = pos - traToRem;
        float dist = pos.norm();
        int level = keys.keyPoints[i].octave;
        dist *= lastKF->scaleFactor[level];
        maxDistsScale.emplace_back(dist);
    }
    const size_t allp4dsize {p4d.size()};
    matchedIdxs = std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>(allp4dsize,std::vector<std::pair<KeyFrame*,std::pair<int, int>>>());
    for ( size_t i {0}; i < allp4dsize; i++)
    {
        matchedIdxs[i].reserve(10);
        std::pair<int,int> keyPos = p4d[i].second;
        matchedIdxs[i].emplace_back(lastKF,keyPos);
    }
}

void LocalMapper::predictKeysPosR(const TrackedKeys& keys, const Eigen::Matrix4d& camPose, const Eigen::Matrix4d& camPoseInv, const std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<std::pair<cv::Point2f, cv::Point2f>>& predPoints)
{
    const Eigen::Matrix4d camPoseInvR = (camPose * zedPtr->extrinsics).inverse();

    const double fxr {zedPtr->cameraRight.fx};
    const double fyr {zedPtr->cameraRight.fy};
    const double cxr {zedPtr->cameraRight.cx};
    const double cyr {zedPtr->cameraRight.cy};
    
    const cv::Point2f noPoint(-1.-1);
    for ( size_t i {0}, end{p4d.size()}; i < end; i ++)
    {
        const Eigen::Vector4d& wp = p4d[i].first;

        Eigen::Vector4d p = camPoseInv * wp;
        Eigen::Vector4d pR = camPoseInvR * wp;

        if ( p(2) <= 0.0 || pR(2) <= 0.0)
        {
            predPoints.emplace_back(noPoint, noPoint);
            continue;
        }

        const double invZ = 1.0f/p(2);
        const double invZR = 1.0f/pR(2);

        double u {fx*p(0)*invZ + cx};
        double v {fy*p(1)*invZ + cy};

        double uR {fxr*pR(0)*invZR + cxr};
        double vR {fyr*pR(1)*invZR + cyr};

        const int w {zedPtr->mWidth};
        const int h {zedPtr->mHeight};

        cv::Point2f predL((float)u, (float)v), predR((float)uR, (float)vR);

        if ( u < 15 || v < 15 || u >= w - 15 || v >= h - 15 )
        {
            predL = noPoint;
        }
        if ( uR < 15 || vR < 15 || uR >= w - 15 || vR >= h - 15 )
        {
            predR = noPoint;
        }

        predPoints.emplace_back(predL, predR);

    }
}

void LocalMapper::triangulateNewPointsR(std::vector<DC_VSLAM::KeyFrame *>& activeKF)
{
    const int kFsize {actvKFMaxSize};
    std::vector<DC_VSLAM::KeyFrame *> actKeyF;
    actKeyF.reserve(kFsize);
    actKeyF = activeKF;
    KeyFrame* lastKF = actKeyF.front();
    const unsigned long lastKFIdx = lastKF->numb;
    std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>> matchedIdxs;

    std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>> p4d;
    std::vector<float> maxDistsScale;
    calcAllMpsOfKFROnlyEst(matchedIdxs, lastKF, kFsize, p4d,maxDistsScale);
    {
    std::vector<KeyFrame*>::const_iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        if ( (*it)->numb == lastKFIdx)
            continue;
        std::vector<std::pair<cv::Point2f, cv::Point2f>> predPoints;
        // predict keys for both right and left camera
        predictKeysPosR(lastKF->keys, (*it)->pose.pose, (*it)->pose.poseInverse, p4d, predPoints);
        fm->matchByProjectionRPredLBA(lastKF, (*it), matchedIdxs, 4, predPoints, maxDistsScale, p4d);
        
    }
    }

    std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>> allProjMatrices;
    allProjMatrices.reserve(2 * actKeyF.size());
    calcProjMatricesR(allProjMatrices, actKeyF);
    std::vector<MapPoint*> pointsToAdd;
    const size_t mpCandSize {matchedIdxs.size()};
    pointsToAdd.resize(mpCandSize,nullptr);
    int newMaPoints {0};
    for ( size_t i{0}; i < mpCandSize; i ++)
    {
        std::vector<std::pair<DC_VSLAM::KeyFrame *, std::pair<int, int>>>& matchesOfPoint = matchedIdxs[i];
        if ((int)matchesOfPoint.size() < minCount)
            continue;
        std::vector<Eigen::Matrix<double, 3, 4>> proj_mat;
        std::vector<Eigen::Vector2d> pointsVec;
        processMatchesR(matchesOfPoint, allProjMatrices, proj_mat, pointsVec);
        Eigen::Vector4d vecCalc = lastKF->pose.getInvPose() * p4d[i].first;
        Eigen::Vector3d vec3d(vecCalc(0), vecCalc(1), vecCalc(2));
        triangulateCeresNew(vec3d, proj_mat, pointsVec, lastKF->pose.pose, true);
        vecCalc(0) = vec3d(0);
        vecCalc(1) = vec3d(1);
        vecCalc(2) = vec3d(2);

        if ( !checkReprojErrNewR(lastKF, vecCalc, matchesOfPoint, proj_mat, pointsVec) )
            continue;

        addMultiViewMapPointsR(vecCalc, matchesOfPoint, pointsToAdd, lastKF, i);
        newMaPoints++;
    }

    addNewMapPoints(lastKF, pointsToAdd, matchedIdxs);
}

void LocalMapper::predictKeysPosRB(const Zed_Camera* zedCam, const Eigen::Matrix4d& camPose, const Eigen::Matrix4d& camPoseInv, const std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<std::pair<cv::Point2f, cv::Point2f>>& predPoints)
{
    const Eigen::Matrix4d camPoseInvR = (camPose * zedCam->extrinsics).inverse();

    const double fx {zedCam->cameraLeft.fx};
    const double fy {zedCam->cameraLeft.fy};
    const double cx {zedCam->cameraLeft.cx};
    const double cy {zedCam->cameraLeft.cy};
    predPoints.reserve(p4d.size());
    const cv::Point2f noPoint(-1.-1);
    for ( size_t i {0}, end{p4d.size()}; i < end; i ++)
    {
        const Eigen::Vector4d& wp = p4d[i].first;

        Eigen::Vector4d p = camPoseInv * wp;
        Eigen::Vector4d pR = camPoseInvR * wp;

        if ( p(2) <= 0.0 || pR(2) <= 0.0)
        {
            predPoints.emplace_back(noPoint, noPoint);
            continue;
        }

        const double invZ = 1.0f/p(2);
        const double invZR = 1.0f/pR(2);

        double u {fx*p(0)*invZ + cx};
        double v {fy*p(1)*invZ + cy};

        double uR {fx*pR(0)*invZR + cx};
        double vR {fy*pR(1)*invZR + cy};

        const int w {zedPtr->mWidth};
        const int h {zedPtr->mHeight};

        cv::Point2f predL((float)u, (float)v), predR((float)uR, (float)vR);

        if ( u < 0 || v < 0 || u >= w || v >= h )
        {
            predL = noPoint;
        }
        if ( uR < 0 || vR < 0 || uR >= w || vR >= h )
        {
            predR = noPoint;
        }

        predPoints.emplace_back(predL, predR);

    }
}

void LocalMapper::triangulateNewPointsRB(const Zed_Camera* zedCam,std::vector<DC_VSLAM::KeyFrame *>& activeKF, const bool back)
{
    const int kFsize {actvKFMaxSize};
    std::vector<DC_VSLAM::KeyFrame *> actKeyF;
    actKeyF.reserve(kFsize);
    actKeyF = activeKF;
    KeyFrame* lastKF = actKeyF.front();
    const unsigned long lastKFIdx = lastKF->numb;
    std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>> matchedIdxs;

    std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>> p4d;
    std::vector<float> maxDistsScale;
    

    calcAllMpsOfKFROnlyEstB(zedCam, matchedIdxs, lastKF, kFsize, p4d,maxDistsScale, back);


    {
    std::vector<KeyFrame*>::const_iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        if ( (*it)->numb == lastKFIdx)
            continue;
        std::vector<std::pair<cv::Point2f, cv::Point2f>> predPoints;
        const Eigen::Matrix4d& camPose = (back) ? (*it)->backPose : (*it)->pose.pose;
        const Eigen::Matrix4d& camPoseInv = (back) ? (*it)->backPoseInv : (*it)->pose.poseInverse;
        predictKeysPosRB(zedCam, camPose, camPoseInv, p4d, predPoints);
        fm->matchByProjectionRPredLBAB(zedCam,lastKF, (*it), matchedIdxs, 4, predPoints, maxDistsScale, p4d, back);
        
    }
    }

    std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>> allProjMatrices;
    allProjMatrices.reserve(2 * actKeyF.size());
    calcProjMatricesRB(zedCam,allProjMatrices, actKeyF, back);
    std::vector<MapPoint*> pointsToAdd;
    const size_t mpCandSize {matchedIdxs.size()};
    pointsToAdd.resize(mpCandSize,nullptr);
    const Eigen::Matrix4d lastKFPoseInv = (back) ? lastKF->backPoseInv : lastKF->pose.poseInverse;
    const Eigen::Matrix4d lastKFPose = (back) ? lastKF->backPose : lastKF->pose.pose;
    for ( size_t i{0}; i < mpCandSize; i ++)
    {
        std::vector<std::pair<DC_VSLAM::KeyFrame *, std::pair<int, int>>>& matchesOfPoint = matchedIdxs[i];
        if ((int)matchesOfPoint.size() < minCount)
            continue;
        std::vector<Eigen::Matrix<double, 3, 4>> proj_mat;
        std::vector<Eigen::Vector2d> pointsVec;
        processMatchesRB(matchesOfPoint, allProjMatrices, proj_mat, pointsVec,back);
        Eigen::Vector4d vecCalc = lastKFPoseInv * p4d[i].first;
        Eigen::Vector3d vec3d(vecCalc(0), vecCalc(1), vecCalc(2));
        triangulateCeresNew(vec3d, proj_mat, pointsVec, lastKFPose, true);
        vecCalc(0) = vec3d(0);
        vecCalc(1) = vec3d(1);
        vecCalc(2) = vec3d(2);

        if ( !checkReprojErrNewRB(lastKF, vecCalc, matchesOfPoint, proj_mat, pointsVec, back) )
            continue;

        addMultiViewMapPointsRB(vecCalc, matchesOfPoint, pointsToAdd, lastKF, i, back);
    }
    addNewMapPointsB(lastKF, pointsToAdd, matchedIdxs,back);
}

bool LocalMapper::checkOutlier(const Eigen::Matrix3d& K, const Eigen::Vector2d& obs, const Eigen::Vector3d posW,const Eigen::Vector3d& tcw, const Eigen::Quaterniond& qcw, const float thresh)
{
    Eigen::Vector3d posC = qcw * posW + tcw;
    if ( posC(2) <= 0 )
        return true;
    Eigen::Vector3d pixel_pose = K * (posC);
    double error_u = obs[0] - pixel_pose[0] / pixel_pose[2];
    double error_v = obs[1] - pixel_pose[1] / pixel_pose[2];
    double error = (error_u * error_u + error_v * error_v);
    if (error > thresh)
        return true;
    else 
        return false;
    
}

bool LocalMapper::checkOutlierR(const Eigen::Matrix3d& K, const Eigen::Matrix3d& qc1c2, const Eigen::Matrix<double,3,1>& tc1c2, const Eigen::Vector2d& obs, const Eigen::Vector3d posW,const Eigen::Vector3d& tcw, const Eigen::Quaterniond& qcw, const float thresh)
{
    Eigen::Vector3d posC = qcw * posW + tcw;
    posC = qc1c2 * posC + tc1c2;
    if ( posC(2) <= 0 )
        return true;
    Eigen::Vector3d pixel_pose = K * (posC);
    double error_u = obs[0] - pixel_pose[0] / pixel_pose[2];
    double error_v = obs[1] - pixel_pose[1] / pixel_pose[2];
    double error = (error_u * error_u + error_v * error_v);
    if (error > thresh)
        return true;
    else 
        return false;
    
}

void LocalMapper::localBAR(std::vector<DC_VSLAM::KeyFrame *>& actKeyF)
{
    std::unordered_map<MapPoint*, Eigen::Vector3d> allMapPoints;
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>> localKFs;
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>> fixedKFs;
    localKFs.reserve(actKeyF.size());
    fixedKFs.reserve(actKeyF.size());
    int blocks {0};
    unsigned long lastActKF {actKeyF.front()->numb};
    bool fixedKF {false};
    std::vector<KeyFrame*>::iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        (*it)->LBAID = lastActKF;
        localKFs[*it] = Converter::Matrix4dToMatrix_7_1((*it)->pose.getInvPose());
        
    }
    for ( it = actKeyF.begin(); it != end; it++)
    {
        if ( (*it)->fixed )
            fixedKF = true;
        std::vector<MapPoint*>::iterator itmp, endmp((*it)->localMapPoints.end());
        for ( itmp = (*it)->localMapPoints.begin(); itmp != endmp; itmp++)
        {
            MapPoint* mp = *itmp;
            if ( !mp )
                continue;
            if ( mp->GetIsOutlier() )
                continue;
            if ( mp->LBAID == (int)lastActKF )
                continue;
            
            std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(mp->kFMatches.end());
            for (kf = mp->kFMatches.begin(); kf != endkf; kf++)
            {
                KeyFrame* kFCand = kf->first;
                if ( !kFCand->keyF || kFCand->numb > lastActKF )
                    continue;
                if (kFCand->LBAID == (int)lastActKF )
                    continue;
                if (localKFs.find(kFCand) == localKFs.end())
                {
                    fixedKFs[kFCand] = Converter::Matrix4dToMatrix_7_1(kFCand->pose.getInvPose());
                    kFCand->LBAID = lastActKF;
                }
                blocks++;
            }
            allMapPoints.insert(std::pair<MapPoint*, Eigen::Vector3d>((*itmp), (*itmp)->getWordPose3d()));
            (*itmp)->LBAID = lastActKF;
        }
        std::vector<MapPoint*>::iterator endmpR((*it)->localMapPointsR.end());
        for ( itmp = (*it)->localMapPointsR.begin(); itmp != endmpR; itmp++)
        {
            MapPoint* mp = *itmp;
            if ( !mp )
                continue;
            if ( mp->GetIsOutlier() )
                continue;
            if ( mp->LBAID == (int)lastActKF )
                continue;
            
            std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(mp->kFMatches.end());
            for (kf = mp->kFMatches.begin(); kf != endkf; kf++)
            {
                KeyFrame* kFCand = kf->first;
                const std::pair<int,int>& keyPos = kf->second;
                if ( keyPos.first >= 0 || keyPos.second < 0 )
                    continue;
                if ( !kFCand->keyF || kFCand->numb > lastActKF )
                    continue;
                if (kFCand->LBAID == (int)lastActKF )
                    continue;
                if (localKFs.find(kFCand) == localKFs.end())
                {
                    fixedKFs[kFCand] = Converter::Matrix4dToMatrix_7_1(kFCand->pose.getInvPose());
                    kFCand->LBAID = lastActKF;
                }
                blocks++;
            }
            allMapPoints.insert(std::pair<MapPoint*, Eigen::Vector3d>((*itmp), (*itmp)->getWordPose3d()));
            (*itmp)->LBAID = lastActKF;
        }
    }
    if ( fixedKFs.size() == 0 && !fixedKF )
    {
        KeyFrame* lastKF = actKeyF.back();
        localKFs.erase(lastKF);
        fixedKFs[lastKF] = Converter::Matrix4dToMatrix_7_1(lastKF->pose.getInvPose());
    }
    std::vector<std::pair<KeyFrame*, MapPoint*>> wrongMatches;
    wrongMatches.reserve(blocks);
    std::vector<bool>mpOutliers;
    mpOutliers.resize(allMapPoints.size());
    bool first = true;
    const Eigen::Matrix3d& K = zedPtr->cameraLeft.intrinsics;
    const Eigen::Matrix4d estimPoseRInv = zedPtr->extrinsics.inverse();
    const Eigen::Matrix3d qc1c2 = estimPoseRInv.block<3,3>(0,0);
    const Eigen::Matrix<double,3,1> tc1c2 = estimPoseRInv.block<3,1>(0,3);
    for (size_t iterations{0}; iterations < 2; iterations++)
    {
    ceres::Problem problem;
    ceres::Manifold* quaternion_local_parameterization = new ceres::EigenQuaternionManifold;
    ceres::LossFunction* loss_function = nullptr;
    if (first)
        loss_function = new ceres::HuberLoss(sqrt(7.815f));
    ceres::ParameterBlockOrdering* ordering = nullptr;
    ordering = new ceres::ParameterBlockOrdering;
    int mpCount {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
    for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
    {
        int timesIn {0};
        bool mpIsOut {true};
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(itmp->first->kFMatches.end());
        for (kf = itmp->first->kFMatches.begin(); kf != endkf; kf++)
        {
            if ( !kf->first->keyF )
                    continue;
            if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && (int)itmp->first->kFMatches.size() < minCount) )
            {
                mpOutliers[mpCount] = true;
                break;
            }
            if ( !wrongMatches.empty() && std::find(wrongMatches.begin(), wrongMatches.end(), std::make_pair(kf->first, itmp->first)) != wrongMatches.end())
            {
                continue;
            }
            if ( itmp->first->GetIsOutlier() )
                break;
            KeyFrame* kftemp = kf->first;
            TrackedKeys& keys = kftemp->keys;
            std::pair<int,int>& keyPos = kf->second;


            if ( kf->first->numb > lastActKF )
            {
                mpIsOut = false;
                continue;
            }
            timesIn ++;
            mpIsOut = false;
            ceres::CostFunction* costf;
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                const cv::KeyPoint& obs = keys.keyPoints[keyPos.first];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {obs.octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustment::Create(K, obs2d, weight);
                close = keys.close[keyPos.first];
            }
            else if ( keyPos.second >= 0 )
            {
                const cv::KeyPoint& obs = keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {obs.octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(K,tc1c2, qc1c2, obs2d, weight);
            }

            ordering->AddElementToGroup(itmp->second.data(), 0);
            if (localKFs.find(kf->first) != localKFs.end())
            {
                ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                if ( kf->first->fixed )
                {
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                }
            }
            else if (fixedKFs.find(kf->first) != fixedKFs.end())
            {
                ordering->AddElementToGroup(fixedKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(fixedKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), fixedKFs[kf->first].block<3,1>(0,0).data(), fixedKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(fixedKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                problem.SetParameterBlockConstant(fixedKFs[kf->first].block<3,1>(0,0).data());
                problem.SetParameterBlockConstant(fixedKFs[kf->first].block<4,1>(3,0).data());
            }
            if ( close )
            {
                if ( keyPos.second < 0 )
                    continue;
                const cv::KeyPoint& obs = kf->first->keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {obs.octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(K,tc1c2, qc1c2, obs2d, weight);

                ordering->AddElementToGroup(itmp->second.data(), 0);
                if (localKFs.find(kf->first) != localKFs.end())
                {
                    ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                    ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                    problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                    problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                    if ( kf->first->fixed )
                    {
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                    }
                }
                else if (fixedKFs.find(kf->first) != fixedKFs.end())
                {
                    ordering->AddElementToGroup(fixedKFs[kf->first].block<3,1>(0,0).data(),1);
                    ordering->AddElementToGroup(fixedKFs[kf->first].block<4,1>(3,0).data(),1);
                    problem.AddResidualBlock(costf, loss_function, itmp->second.data(), fixedKFs[kf->first].block<3,1>(0,0).data(), fixedKFs[kf->first].block<4,1>(3,0).data());
                    problem.SetManifold(fixedKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                    problem.SetParameterBlockConstant(fixedKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(fixedKFs[kf->first].block<4,1>(3,0).data());
                }
            }

        }
        if ( mpIsOut )
            mpOutliers[mpCount] = true;
    }
    
    ceres::Solver::Options options;
    options.linear_solver_ordering.reset(ordering);
    options.num_threads = 1;
    options.max_num_iterations = 10;
    if ( first )
        options.max_num_iterations = 5;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.use_explicit_schur_complement = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::vector<std::pair<KeyFrame*, MapPoint*>> emptyVec;
    wrongMatches.swap(emptyVec);

    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator allmp, allmpend(allMapPoints.end());
    for (allmp = allMapPoints.begin(); allmp != allmpend; allmp ++)
    {
        MapPoint* mp = allmp->first;
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(mp->kFMatches.end());
        for (kf = mp->kFMatches.begin(); kf != endkf; kf++)
        {
            KeyFrame* kfCand = kf->first;
            std::pair<int,int>& keyPos = kf->second;
            if ( localKFs.find(kfCand) == localKFs.end() )
                continue;
            cv::KeyPoint kp;
            bool right {false};
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                kp = kfCand->keys.keyPoints[keyPos.first];
                close = kfCand->keys.close[keyPos.first];
            }
            else if ( keyPos.second >= 0 )
            {
                kp = kfCand->keys.rightKeyPoints[keyPos.second];
                right = true;
            }
            Eigen::Vector2d obs( (double)kp.pt.x, (double)kp.pt.y);
            const int oct = kp.octave;
            const double weight = (double)kfCand->sigmaFactor[oct];
            Eigen::Vector3d tcw = localKFs[kfCand].block<3, 1>(0, 0);
            Eigen::Vector4d q_xyzw = localKFs[kfCand].block<4, 1>(3, 0);
            Eigen::Quaterniond qcw(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
            bool outlier {false};
            if ( right )
                outlier = checkOutlierR(K,qc1c2, tc1c2, obs, allmp->second, tcw, qcw, reprjThreshold * weight);
            else
                outlier = checkOutlier(K, obs, allmp->second, tcw, qcw, reprjThreshold * weight);

            if ( outlier )
                wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
            else
            {
                if ( close )
                {
                    if ( keyPos.second < 0 )
                        continue;
                    cv::KeyPoint kpR = kfCand->keys.rightKeyPoints[keyPos.second];
                    const int octR = kpR.octave;
                    const double weightR = (double)kfCand->sigmaFactor[octR];
                    Eigen::Vector2d obsr( (double)kpR.pt.x, (double)kpR.pt.y);
                    bool outlierR = checkOutlierR(K,qc1c2, tc1c2, obsr, allmp->second, tcw, qcw, reprjThreshold * weightR);
                    if ( outlierR )
                    {
                        wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
                    }
                }
            }

        }
    }
    first = false;
    }
    std::lock_guard<std::mutex> lock(map->mapMutex);

    if ( !wrongMatches.empty() )
    {
        for (size_t wM {0}, endwM {wrongMatches.size()}; wM < endwM; wM ++)
        {
            KeyFrame* kF = wrongMatches[wM].first;
            MapPoint* mp = wrongMatches[wM].second;
            const std::pair<int,int>& keyPos = mp->kFMatches.at(kF);
            kF->eraseMPConnection(keyPos);
            mp->eraseKFConnection(kF);
        }
    }


    
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>::iterator localkf, endlocalkf(localKFs.end());
    for ( localkf = localKFs.begin(); localkf != endlocalkf; localkf++)
    {
        localkf->first->pose.setInvPose(Converter::Matrix_7_1_ToMatrix4d(localkf->second));
        localkf->first->LBA = true;
    }

    int mpCount {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
    for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
    {
        if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && (int)itmp->first->kFMatches.size() < minCount) )
            itmp->first->SetIsOutlier(true);
        else
        {
            itmp->first->updatePos(itmp->second, zedPtr);
        }
    }

    

    map->endLBAIdx = actKeyF.front()->numb;
    map->keyFrameAdded = false;
    map->LBADone = true;
    
}

void LocalMapper::loopClosureR(std::vector<DC_VSLAM::KeyFrame *>& actKeyF)
{
    std::cout << "Loop Closure Detected! Starting Optimization.." << std::endl;
    Timer LC("Loop Closure ");
    std::unordered_map<MapPoint*, Eigen::Vector3d> allMapPoints;
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>> localKFs;
    localKFs.reserve(actKeyF.size());
    const unsigned long lastActKF {actKeyF.front()->numb};
    KeyFrame* lCCand = actKeyF.front();
    localKFs[lCCand] = Converter::Matrix4dToMatrix_7_1(map->LCPose.inverse());
    lCCand->fixed = true;
    std::vector<KeyFrame*>::iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        (*it)->LCID = lastActKF;
        if ( (*it)->numb == lastActKF )
            continue;
        localKFs[*it] = Converter::Matrix4dToMatrix_7_1((*it)->pose.getInvPose());
        
    }
    for ( it = actKeyF.begin(); it != end; it++)
    {
        std::vector<MapPoint*>::iterator itmp, endmp((*it)->localMapPoints.end());
        for ( itmp = (*it)->localMapPoints.begin(); itmp != endmp; itmp++)
        {
            MapPoint* mp = *itmp;
            if ( !mp )
                continue;
            if ( mp->GetIsOutlier() )
                continue;
            if ( mp->LCID == (int)lastActKF )
                continue;
            
            allMapPoints.insert(std::pair<MapPoint*, Eigen::Vector3d>((*itmp), (*itmp)->getWordPose3d()));
            (*itmp)->LCID = lastActKF;
        }
        std::vector<MapPoint*>::iterator endmpR((*it)->localMapPointsR.end());
        for ( itmp = (*it)->localMapPointsR.begin(); itmp != endmpR; itmp++)
        {
            MapPoint* mp = *itmp;
            if ( !mp )
                continue;
            if ( mp->GetIsOutlier() )
                continue;
            if ( mp->LCID == (int)lastActKF )
                continue;

            allMapPoints.insert(std::pair<MapPoint*, Eigen::Vector3d>((*itmp), (*itmp)->getWordPose3d()));
            (*itmp)->LCID = lastActKF;
        }
    }

    std::vector<std::pair<KeyFrame*, MapPoint*>> wrongMatches;
    wrongMatches.reserve(allMapPoints.size());
    std::vector<bool>mpOutliers;
    mpOutliers.resize(allMapPoints.size());
    bool first = true;
    const Eigen::Matrix3d& K = zedPtr->cameraLeft.intrinsics;
    const Eigen::Matrix4d estimPoseRInv = zedPtr->extrinsics.inverse();
    const Eigen::Matrix3d qc1c2 = estimPoseRInv.block<3,3>(0,0);
    const Eigen::Matrix<double,3,1> tc1c2 = estimPoseRInv.block<3,1>(0,3);
    for (size_t iterations{0}; iterations < 2; iterations++)
    {
    ceres::Problem problem;
    ceres::Manifold* quaternion_local_parameterization = new ceres::EigenQuaternionManifold;
    ceres::LossFunction* loss_function = nullptr;
    if (first)
        loss_function = new ceres::HuberLoss(sqrt(7.815f));
    ceres::ParameterBlockOrdering* ordering = nullptr;
    ordering = new ceres::ParameterBlockOrdering;
    int mpCount {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
    for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
    {
        int timesIn {0};
        bool mpIsOut {true};
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(itmp->first->kFMatches.end());
        for (kf = itmp->first->kFMatches.begin(); kf != endkf; kf++)
        {
            if ( !kf->first->keyF )
                    continue;
            if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && (int)itmp->first->kFMatches.size() < minCount) )
                break;
            if ( !wrongMatches.empty() && std::find(wrongMatches.begin(), wrongMatches.end(), std::make_pair(kf->first, itmp->first)) != wrongMatches.end())
            {
                continue;
            }
            if ( itmp->first->GetIsOutlier() )
                break;
            KeyFrame* kftemp = kf->first;
            TrackedKeys& keys = kftemp->keys;
            std::pair<int,int>& keyPos = kf->second;


            if ( kf->first->numb > lastActKF )
            {
                mpIsOut = false;
                continue;
            }
            timesIn ++;
            mpIsOut = false;
            ceres::CostFunction* costf;
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                const cv::KeyPoint& obs = keys.keyPoints[keyPos.first];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {obs.octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustment::Create(K, obs2d, weight);
                close = keys.close[keyPos.first];
            }
            else if ( keyPos.second >= 0 )
            {
                const cv::KeyPoint& obs = keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {obs.octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(K,tc1c2, qc1c2, obs2d, weight);
            }

            ordering->AddElementToGroup(itmp->second.data(), 0);
            if (localKFs.find(kf->first) != localKFs.end())
            {
                ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                if ( kf->first->fixed )
                {
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                }
            }
            else
                continue;
            if ( close )
            {
                if ( keyPos.second < 0 )
                    continue;
                const cv::KeyPoint& obs = kf->first->keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {obs.octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(K,tc1c2, qc1c2, obs2d, weight);

                ordering->AddElementToGroup(itmp->second.data(), 0);
                if (localKFs.find(kf->first) != localKFs.end())
                {
                    ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                    ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                    problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                    problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                    if ( kf->first->fixed )
                    {
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                    }
                }
                else
                    continue;
            }

        }
        if ( mpIsOut )
            mpOutliers[mpCount] = true;
    }
    
    ceres::Solver::Options options;
    options.linear_solver_ordering.reset(ordering);
    options.num_threads = 8;
    options.max_num_iterations = 45;
    if ( first )
        options.max_num_iterations = 5;
    options.linear_solver_type = ceres::SPARSE_SCHUR;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::vector<std::pair<KeyFrame*, MapPoint*>> emptyVec;
    wrongMatches.swap(emptyVec);
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator allmp, allmpend(allMapPoints.end());
    for (allmp = allMapPoints.begin(); allmp != allmpend; allmp ++)
    {
        MapPoint* mp = allmp->first;
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(mp->kFMatches.end());
        for (kf = mp->kFMatches.begin(); kf != endkf; kf++)
        {
            KeyFrame* kfCand = kf->first;
            std::pair<int,int>& keyPos = kf->second;
            if ( localKFs.find(kfCand) == localKFs.end() )
                continue;
            cv::KeyPoint kp;
            bool right {false};
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                kp = kfCand->keys.keyPoints[keyPos.first];
                close = kfCand->keys.close[keyPos.first];
            }
            else if ( keyPos.second >= 0 )
            {
                kp = kfCand->keys.rightKeyPoints[keyPos.second];
                right = true;
            }
            Eigen::Vector2d obs( (double)kp.pt.x, (double)kp.pt.y);
            const int oct = kp.octave;
            const double weight = (double)kfCand->sigmaFactor[oct];
            Eigen::Vector3d tcw = localKFs[kfCand].block<3, 1>(0, 0);
            Eigen::Vector4d q_xyzw = localKFs[kfCand].block<4, 1>(3, 0);
            Eigen::Quaterniond qcw(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
            bool outlier {false};
            if ( right )
                outlier = checkOutlierR(K,qc1c2, tc1c2, obs, allmp->second, tcw, qcw, reprjThreshold * weight);
            else
                outlier = checkOutlier(K, obs, allmp->second, tcw, qcw, reprjThreshold * weight);

            if ( outlier )
                wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
            else
            {
                if ( close )
                {
                    if ( keyPos.second < 0 )
                        continue;
                    cv::KeyPoint kpR = kfCand->keys.rightKeyPoints[keyPos.second];
                    const int octR = kpR.octave;
                    const double weightR = (double)kfCand->sigmaFactor[octR];
                    Eigen::Vector2d obsr( (double)kpR.pt.x, (double)kpR.pt.y);
                    bool outlierR = checkOutlierR(K,qc1c2, tc1c2, obsr, allmp->second, tcw, qcw, reprjThreshold * weightR);
                    if ( outlierR )
                    {
                        wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
                    }
                }
            }

        }
    }
    first = false;
    }
    std::lock_guard<std::mutex> lock(map->mapMutex);

    if ( !wrongMatches.empty() )
    {
        for (size_t wM {0}, endwM {wrongMatches.size()}; wM < endwM; wM ++)
        {
            KeyFrame* kF = wrongMatches[wM].first;
            MapPoint* mp = wrongMatches[wM].second;
            const std::pair<int,int>& keyPos = mp->kFMatches.at(kF);
            kF->eraseMPConnection(keyPos);
            mp->eraseKFConnection(kF);
        }
    }


    
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>::iterator localkf, endlocalkf(localKFs.end());
    for ( localkf = localKFs.begin(); localkf != endlocalkf; localkf++)
    {
        localkf->first->pose.setInvPose(Converter::Matrix_7_1_ToMatrix4d(localkf->second));
    }

    int mpCount {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
    for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
    {
        if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && (int)itmp->first->kFMatches.size() < minCount) )
            itmp->first->SetIsOutlier(true);
        else
        {
            itmp->first->updatePos(itmp->second, zedPtr);
        }
    }

    

    map->endLCIdx = actKeyF.front()->numb;
    map->LCDone = true;
    map->LCStart = false;
    map->aprilTagDetected = false;
    std::cout << "Loop Closure Optimization Finished!" << std::endl;
}

void LocalMapper::insertMPsForLBA(std::vector<MapPoint*>& localMapPoints, const std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>& localKFs,std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>& fixedKFs, std::unordered_map<MapPoint*, Eigen::Vector3d>& allMapPoints, const unsigned long lastActKF, int& blocks, const bool back)
{
    std::vector<MapPoint*>::iterator itmp, endmp(localMapPoints.end());
    for ( itmp = localMapPoints.begin(); itmp != endmp; itmp++)
    {
        MapPoint* mp = *itmp;
        if ( !mp )
            continue;
        if ( mp->GetIsOutlier() )
            continue;
        if ( mp->LBAID == (int)lastActKF )
            continue;
        
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf = (back) ? mp->kFMatchesB.begin() : mp->kFMatches.begin();
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator endkf = (back) ? mp->kFMatchesB.end() : mp->kFMatches.end();
        for (; kf != endkf; kf++)
        {
            KeyFrame* kFCand = kf->first;
            if ( !kFCand->keyF || kFCand->numb > lastActKF )
                continue;
            if (kFCand->LBAID == (int)lastActKF )
                continue;
            if (localKFs.find(kFCand) == localKFs.end())
            {
                fixedKFs[kFCand] = Converter::Matrix4dToMatrix_7_1(kFCand->pose.getInvPose());
                kFCand->LBAID = lastActKF;
            }
            blocks++;
        }
        allMapPoints.insert(std::pair<MapPoint*, Eigen::Vector3d>((*itmp), (*itmp)->getWordPose3d()));
        (*itmp)->LBAID = lastActKF;
    }
}

void LocalMapper::insertMPsForLC(std::vector<MapPoint*>& localMapPoints, const std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>& localKFs, std::unordered_map<MapPoint*, Eigen::Vector3d>& allMapPoints, const unsigned long lastActKF, int& blocks, const bool back)
{
    std::vector<MapPoint*>::iterator itmp, endmp(localMapPoints.end());
    for ( itmp = localMapPoints.begin(); itmp != endmp; itmp++)
    {
        MapPoint* mp = *itmp;
        if ( !mp )
            continue;
        if ( mp->GetIsOutlier() )
            continue;
        if ( mp->LCID == (int)lastActKF )
            continue;

        allMapPoints.insert(std::pair<MapPoint*, Eigen::Vector3d>((*itmp), (*itmp)->getWordPose3d()));
        (*itmp)->LCID = lastActKF;
    }
}

void LocalMapper::localBARB(std::vector<DC_VSLAM::KeyFrame *>& actKeyF)
{
    std::unordered_map<MapPoint*, Eigen::Vector3d> allMapPoints;
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>> localKFs;
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>> fixedKFs;
    localKFs.reserve(actKeyF.size());
    fixedKFs.reserve(actKeyF.size());
    int blocks {0};
    unsigned long lastActKF {actKeyF.front()->numb};
    bool fixedKF {false};
    std::vector<KeyFrame*>::iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        (*it)->LBAID = lastActKF;
        localKFs[*it] = Converter::Matrix4dToMatrix_7_1((*it)->pose.getInvPose());
        
    }
    for ( it = actKeyF.begin(); it != end; it++)
    {
        if ( (*it)->fixed )
            fixedKF = true;
        insertMPsForLBA((*it)->localMapPoints,localKFs, fixedKFs, allMapPoints, lastActKF, blocks, false);
        insertMPsForLBA((*it)->localMapPointsR,localKFs, fixedKFs, allMapPoints, lastActKF, blocks, false);
        insertMPsForLBA((*it)->localMapPointsB,localKFs, fixedKFs, allMapPoints, lastActKF, blocks, true);
        insertMPsForLBA((*it)->localMapPointsRB,localKFs, fixedKFs, allMapPoints, lastActKF, blocks, true);
    }
    if ( fixedKFs.size() == 0 && !fixedKF )
    {
        KeyFrame* lastKF = actKeyF.back();
        localKFs.erase(lastKF);
        fixedKFs[lastKF] = Converter::Matrix4dToMatrix_7_1(lastKF->pose.getInvPose());
    }
    std::vector<std::pair<KeyFrame*, MapPoint*>> wrongMatches;
    wrongMatches.reserve(blocks);
    std::vector<bool>mpOutliers;
    mpOutliers.resize(allMapPoints.size());
    bool first = true;
    const Eigen::Matrix3d& K = zedPtr->cameraLeft.intrinsics;
    const Eigen::Matrix3d& KB = zedPtrB->cameraLeft.intrinsics;
    const Eigen::Matrix4d estimPoseRInv = zedPtr->extrinsics.inverse();
    const Eigen::Matrix3d qcr = estimPoseRInv.block<3,3>(0,0);
    const Eigen::Matrix<double,3,1> tcr = estimPoseRInv.block<3,1>(0,3);

    const Eigen::Matrix4d estimPoseBInv = zedPtr->TCamToCamInv;
    const Eigen::Matrix4d estimPoseBRInv = zedPtrB->extrinsics.inverse() * estimPoseBInv;
    const Eigen::Matrix3d qc1c2B = estimPoseBInv.block<3,3>(0,0);
    const Eigen::Matrix<double,3,1> tc1c2B = estimPoseBInv.block<3,1>(0,3);
    const Eigen::Matrix3d qc1c2BR = estimPoseBRInv.block<3,3>(0,0);
    const Eigen::Matrix<double,3,1> tc1c2BR = estimPoseBRInv.block<3,1>(0,3);

    for (size_t iterations{0}; iterations < 2; iterations++)
    {
    ceres::Problem problem;
    ceres::Manifold* quaternion_local_parameterization = new ceres::EigenQuaternionManifold;
    ceres::LossFunction* loss_function = nullptr;
    if (first)
        loss_function = new ceres::HuberLoss(sqrt(7.815f));
    ceres::ParameterBlockOrdering* ordering = nullptr;
    ordering = new ceres::ParameterBlockOrdering;
    int mpCount {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
    for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
    {
        int timesIn {0};
        bool mpIsOut {true};
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(itmp->first->kFMatches.end());
        for (kf = itmp->first->kFMatches.begin(); kf != endkf; kf++)
        {
            if ( !kf->first->keyF )
                    continue;
            if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && (int)itmp->first->kFMatches.size() < minCount) )
                break;
            if ( !wrongMatches.empty() && std::find(wrongMatches.begin(), wrongMatches.end(), std::make_pair(kf->first, itmp->first)) != wrongMatches.end())
            {
                continue;
            }
            if ( itmp->first->GetIsOutlier() )
                break;
            KeyFrame* kftemp = kf->first;
            TrackedKeys& keys = kftemp->keys;
            std::pair<int,int>& keyPos = kf->second;


            if ( kf->first->numb > lastActKF )
            {
                mpIsOut = false;
                continue;
            }
            timesIn ++;
            mpIsOut = false;
            ceres::CostFunction* costf;
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                const cv::KeyPoint& obs = keys.keyPoints[keyPos.first];
                close = keys.close[keyPos.first];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.keyPoints[keyPos.first].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustment::Create(K, obs2d, weight);
            }
            else if ( keyPos.second >= 0 )
            {
                const cv::KeyPoint& obs = keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.rightKeyPoints[keyPos.second].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(K,tcr, qcr, obs2d, weight);
            }

            ordering->AddElementToGroup(itmp->second.data(), 0);
            if (localKFs.find(kf->first) != localKFs.end())
            {
                ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                if ( kf->first->fixed )
                {
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                }
            }
            else if (fixedKFs.find(kf->first) != fixedKFs.end())
            {
                ordering->AddElementToGroup(fixedKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(fixedKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), fixedKFs[kf->first].block<3,1>(0,0).data(), fixedKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(fixedKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                problem.SetParameterBlockConstant(fixedKFs[kf->first].block<3,1>(0,0).data());
                problem.SetParameterBlockConstant(fixedKFs[kf->first].block<4,1>(3,0).data());
            }
            if ( close )
            {
                if ( keyPos.second < 0 )
                    continue;
                const cv::KeyPoint& obs = keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.rightKeyPoints[keyPos.second].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(K,tcr, qcr, obs2d, weight);
                if (localKFs.find(kf->first) != localKFs.end())
                {
                    ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                    ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                    problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                    problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                    if ( kf->first->fixed )
                    {
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                    }
                }
                else if (fixedKFs.find(kf->first) != fixedKFs.end())
                {
                    ordering->AddElementToGroup(fixedKFs[kf->first].block<3,1>(0,0).data(),1);
                    ordering->AddElementToGroup(fixedKFs[kf->first].block<4,1>(3,0).data(),1);
                    problem.AddResidualBlock(costf, loss_function, itmp->second.data(), fixedKFs[kf->first].block<3,1>(0,0).data(), fixedKFs[kf->first].block<4,1>(3,0).data());
                    problem.SetManifold(fixedKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                    problem.SetParameterBlockConstant(fixedKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(fixedKFs[kf->first].block<4,1>(3,0).data());
                }
            }

        }
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator endkfB(itmp->first->kFMatchesB.end());
        for (kf = itmp->first->kFMatchesB.begin(); kf != endkfB; kf++)
        {
            if ( !kf->first->keyF )
                    continue;
            if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && (int)itmp->first->kFMatchesB.size() < minCount) )
                break;
            if ( !wrongMatches.empty() && std::find(wrongMatches.begin(), wrongMatches.end(), std::make_pair(kf->first, itmp->first)) != wrongMatches.end())
            {
                continue;
            }
            if ( itmp->first->GetIsOutlier() )
                break;
            KeyFrame* kftemp = kf->first;
            TrackedKeys& keys = kftemp->keysB;
            std::pair<int,int>& keyPos = kf->second;


            if ( kf->first->numb > lastActKF )
            {
                mpIsOut = false;
                continue;
            }
            timesIn ++;
            mpIsOut = false;
            ceres::CostFunction* costf;
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                const cv::KeyPoint& obs = keys.keyPoints[keyPos.first];
                close = keys.close[keyPos.first];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.keyPoints[keyPos.first].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(KB,tc1c2B, qc1c2B, obs2d, weight);
            }
            else if ( keyPos.second >= 0 )
            {
                const cv::KeyPoint& obs = keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.rightKeyPoints[keyPos.second].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(KB,tc1c2BR, qc1c2BR, obs2d, weight);
            }

            ordering->AddElementToGroup(itmp->second.data(), 0);
            if (localKFs.find(kf->first) != localKFs.end())
            {
                ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                if ( kf->first->fixed )
                {
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                }
            }
            else if (fixedKFs.find(kf->first) != fixedKFs.end())
            {
                ordering->AddElementToGroup(fixedKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(fixedKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), fixedKFs[kf->first].block<3,1>(0,0).data(), fixedKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(fixedKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                problem.SetParameterBlockConstant(fixedKFs[kf->first].block<3,1>(0,0).data());
                problem.SetParameterBlockConstant(fixedKFs[kf->first].block<4,1>(3,0).data());
            }
            if ( close )
            {
                
                if ( keyPos.second < 0 )
                    continue;
                const cv::KeyPoint& obs = keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.rightKeyPoints[keyPos.second].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(KB,tc1c2BR, qc1c2BR, obs2d, weight);
                if (localKFs.find(kf->first) != localKFs.end())
                {
                    ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                    ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                    problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                    problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                    if ( kf->first->fixed )
                    {
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                    }
                }
                else if (fixedKFs.find(kf->first) != fixedKFs.end())
                {
                    ordering->AddElementToGroup(fixedKFs[kf->first].block<3,1>(0,0).data(),1);
                    ordering->AddElementToGroup(fixedKFs[kf->first].block<4,1>(3,0).data(),1);
                    problem.AddResidualBlock(costf, loss_function, itmp->second.data(), fixedKFs[kf->first].block<3,1>(0,0).data(), fixedKFs[kf->first].block<4,1>(3,0).data());
                    problem.SetManifold(fixedKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                    problem.SetParameterBlockConstant(fixedKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(fixedKFs[kf->first].block<4,1>(3,0).data());
                }
            }
        }
        if ( mpIsOut )
            mpOutliers[mpCount] = true;
    }
    
    ceres::Solver::Options options;
    options.linear_solver_ordering.reset(ordering);
    options.num_threads = 1;
    options.max_num_iterations = 10;
    if ( first )
        options.max_num_iterations = 5;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.use_explicit_schur_complement = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::vector<std::pair<KeyFrame*, MapPoint*>> emptyVec;
    wrongMatches.swap(emptyVec);


    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator allmp, allmpend(allMapPoints.end());
    for (allmp = allMapPoints.begin(); allmp != allmpend; allmp ++)
    {
        MapPoint* mp = allmp->first;
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(mp->kFMatches.end());
        for (kf = mp->kFMatches.begin(); kf != endkf; kf++)
        {
            KeyFrame* kfCand = kf->first;
            std::pair<int,int>& keyPos = kf->second;
            if ( localKFs.find(kfCand) == localKFs.end() )
                continue;
            cv::KeyPoint kp;
            bool right {false};
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                kp = kfCand->keys.keyPoints[keyPos.first];
                close = kfCand->keys.close[keyPos.first];
            }
            else if ( keyPos.second >= 0 )
            {
                kp = kfCand->keys.rightKeyPoints[keyPos.second];
                right = true;
            }
            Eigen::Vector2d obs( (double)kp.pt.x, (double)kp.pt.y);
            const int oct = kp.octave;
            const double weight = (double)kfCand->sigmaFactor[oct];
            Eigen::Vector3d tcw = localKFs[kfCand].block<3, 1>(0, 0);
            Eigen::Vector4d q_xyzw = localKFs[kfCand].block<4, 1>(3, 0);
            Eigen::Quaterniond qcw(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
            bool outlier {false};
            if ( right )
                outlier = checkOutlierR(K,qcr, tcr, obs, allmp->second, tcw, qcw, reprjThreshold * weight);
            else
                outlier = checkOutlier(K, obs, allmp->second, tcw, qcw, reprjThreshold * weight);

            if ( outlier )
                wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
            else
            {
                if ( close )
                {
                    if ( keyPos.second < 0 )
                        continue;
                    cv::KeyPoint kpR = kfCand->keys.rightKeyPoints[keyPos.second];
                    const int octR = kpR.octave;
                    const double weightR = (double)kfCand->sigmaFactor[octR];
                    Eigen::Vector2d obsr( (double)kpR.pt.x, (double)kpR.pt.y);
                    bool outlierR = checkOutlierR(K,qcr, tcr, obsr, allmp->second, tcw, qcw, reprjThreshold * weightR);
                    if ( outlierR )
                    {
                        wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
                    }
                }
            }

        }
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator endkfB(mp->kFMatchesB.end());
        for (kf = mp->kFMatchesB.begin(); kf != endkfB; kf++)
        {
            KeyFrame* kfCand = kf->first;
            std::pair<int,int>& keyPos = kf->second;
            if ( localKFs.find(kfCand) == localKFs.end() )
                continue;
            cv::KeyPoint kp;
            bool right {false};
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                kp = kfCand->keysB.keyPoints[keyPos.first];
                close = kfCand->keysB.close[keyPos.first];
            }
            else if ( keyPos.second >= 0 )
            {
                kp = kfCand->keysB.rightKeyPoints[keyPos.second];
                right = true;
            }
            Eigen::Vector2d obs( (double)kp.pt.x, (double)kp.pt.y);
            const int oct = kp.octave;
            const double weight = (double)kfCand->sigmaFactor[oct];
            Eigen::Vector3d tcw = localKFs[kfCand].block<3, 1>(0, 0);
            Eigen::Vector4d q_xyzw = localKFs[kfCand].block<4, 1>(3, 0);
            Eigen::Quaterniond qcw(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
            bool outlier {false};
            if ( right )
                outlier = checkOutlierR(KB,qc1c2BR, tc1c2BR, obs, allmp->second, tcw, qcw, reprjThreshold * weight);
            else
                outlier = checkOutlierR(KB,qc1c2B, tc1c2B, obs, allmp->second, tcw, qcw, reprjThreshold * weight);

            if ( outlier )
                wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
            else
            {
                if ( close )
                {
                    if ( keyPos.second < 0 )
                        continue;
                    cv::KeyPoint kpR = kfCand->keysB.rightKeyPoints[keyPos.second];
                    Eigen::Vector2d obsr( (double)kpR.pt.x, (double)kpR.pt.y);
                    const int octR = kpR.octave;
                    const double weightR = (double)kfCand->sigmaFactor[octR];
                    bool outlierR = checkOutlierR(KB,qc1c2BR, tc1c2BR, obsr, allmp->second, tcw, qcw, reprjThreshold * weightR);
                    if ( outlierR )
                    {
                        wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
                    }
                }
            }

        }
    }
    first = false;
    }
    std::lock_guard<std::mutex> lock(map->mapMutex);

    if ( !wrongMatches.empty() )
    {
        for (size_t wM {0}, endwM {wrongMatches.size()}; wM < endwM; wM ++)
        {
            KeyFrame* kF = wrongMatches[wM].first;
            MapPoint* mp = wrongMatches[wM].second;
            if ( mp->kFMatches.size() > 0 )
            {
                const std::pair<int,int>& keyPos = mp->kFMatches.at(kF);
                kF->eraseMPConnection(keyPos);
                mp->eraseKFConnection(kF);
            }
            if ( mp->kFMatchesB.size() > 0 )
            {
                const std::pair<int,int>& keyPos = mp->kFMatchesB.at(kF);
                kF->eraseMPConnectionB(keyPos);
                mp->eraseKFConnectionB(kF);
            }
        }
    }


    
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>::iterator localkf, endlocalkf(localKFs.end());
    for ( localkf = localKFs.begin(); localkf != endlocalkf; localkf++)
    {
        localkf->first->pose.setInvPose(Converter::Matrix_7_1_ToMatrix4d(localkf->second));
        localkf->first->setBackPose(localkf->first->pose.pose * zedPtr->TCamToCam);
        localkf->first->LBA = true;
    }

    int mpCount {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
    for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
    {
        if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && (int)itmp->first->kFMatches.size() < minCount && (int)itmp->first->kFMatchesB.size() < minCount) )
            itmp->first->SetIsOutlier(true);
        else
        {
            itmp->first->updatePos(itmp->second, zedPtr);
        }
    }

    map->endLBAIdx = actKeyF.front()->numb;
    map->keyFrameAdded = false;
    map->LBADone = true;
}

void LocalMapper::loopClosureRB(std::vector<DC_VSLAM::KeyFrame *>& actKeyF)
{
    std::cout << "Loop Closure Detected! Starting Optimization.." << std::endl;
    Timer LC("Loop Closure ");
    std::unordered_map<MapPoint*, Eigen::Vector3d> allMapPoints;
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>> localKFs;
    localKFs.reserve(actKeyF.size());
    int blocks {0};
    unsigned long lastActKF {actKeyF.front()->numb};
    KeyFrame* lCCand = actKeyF.front();
    localKFs[lCCand] = Converter::Matrix4dToMatrix_7_1(map->LCPose.inverse());
    lCCand->fixed = true;
    std::vector<KeyFrame*>::iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        (*it)->LCID = lastActKF;
        if ( (*it)->numb == lastActKF )
            continue;
        localKFs[*it] = Converter::Matrix4dToMatrix_7_1((*it)->pose.getInvPose());
        
    }
    for ( it = actKeyF.begin(); it != end; it++)
    {
        insertMPsForLC((*it)->localMapPoints,localKFs, allMapPoints, lastActKF, blocks, false);
        insertMPsForLC((*it)->localMapPointsR,localKFs, allMapPoints, lastActKF, blocks, false);
        insertMPsForLC((*it)->localMapPointsB,localKFs, allMapPoints, lastActKF, blocks, true);
        insertMPsForLC((*it)->localMapPointsRB,localKFs, allMapPoints, lastActKF, blocks, true);
    }
    std::vector<std::pair<KeyFrame*, MapPoint*>> wrongMatches;
    wrongMatches.reserve(blocks);
    std::vector<bool>mpOutliers;
    mpOutliers.resize(allMapPoints.size());
    bool first = true;
    const Eigen::Matrix3d& K = zedPtr->cameraLeft.intrinsics;
    const Eigen::Matrix3d& KB = zedPtrB->cameraLeft.intrinsics;
    const Eigen::Matrix4d estimPoseRInv = zedPtr->extrinsics.inverse();
    const Eigen::Matrix3d qcr = estimPoseRInv.block<3,3>(0,0);
    const Eigen::Matrix<double,3,1> tcr = estimPoseRInv.block<3,1>(0,3);

    const Eigen::Matrix4d estimPoseBInv = zedPtr->TCamToCamInv;
    const Eigen::Matrix4d estimPoseBRInv = zedPtrB->extrinsics.inverse() * estimPoseBInv;
    const Eigen::Matrix3d qc1c2B = estimPoseBInv.block<3,3>(0,0);
    const Eigen::Matrix<double,3,1> tc1c2B = estimPoseBInv.block<3,1>(0,3);
    const Eigen::Matrix3d qc1c2BR = estimPoseBRInv.block<3,3>(0,0);
    const Eigen::Matrix<double,3,1> tc1c2BR = estimPoseBRInv.block<3,1>(0,3);

    for (size_t iterations{0}; iterations < 1; iterations++)
    {
    ceres::Problem problem;
    ceres::Manifold* quaternion_local_parameterization = new ceres::EigenQuaternionManifold;
    ceres::LossFunction* loss_function = nullptr;
    if (first)
        loss_function = new ceres::HuberLoss(sqrt(7.815f));
    ceres::ParameterBlockOrdering* ordering = nullptr;
    ordering = new ceres::ParameterBlockOrdering;
    int mpCount {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
    for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
    {
        int timesIn {0};
        bool mpIsOut {true};
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(itmp->first->kFMatches.end());
        for (kf = itmp->first->kFMatches.begin(); kf != endkf; kf++)
        {
            if ( !kf->first->keyF )
                    continue;
            if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && (int)itmp->first->kFMatches.size() < minCount) )
                break;
            if ( itmp->first->GetIsOutlier() )
                break;
            KeyFrame* kftemp = kf->first;
            TrackedKeys& keys = kftemp->keys;
            std::pair<int,int>& keyPos = kf->second;

            if ( kf->first->numb > lastActKF )
            {
                mpIsOut = false;
                continue;
            }
            timesIn ++;
            mpIsOut = false;
            ceres::CostFunction* costf;
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                const cv::KeyPoint& obs = keys.keyPoints[keyPos.first];
                close = keys.close[keyPos.first];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.keyPoints[keyPos.first].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustment::Create(K, obs2d, weight);
            }
            else if ( keyPos.second >= 0 )
            {
                const cv::KeyPoint& obs = keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.rightKeyPoints[keyPos.second].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(K,tcr, qcr, obs2d, weight);
            }

            ordering->AddElementToGroup(itmp->second.data(), 0);
            if (localKFs.find(kf->first) != localKFs.end())
            {
                ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                if ( kf->first->fixed )
                {
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                }
            }
            else
                continue;
            if ( close )
            {
                if ( keyPos.second < 0 )
                    continue;
                const cv::KeyPoint& obs = keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.rightKeyPoints[keyPos.second].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(K,tcr, qcr, obs2d, weight);
                if (localKFs.find(kf->first) != localKFs.end())
                {
                    ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                    ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                    problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                    problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                    if ( kf->first->fixed )
                    {
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                    }
                }
                else 
                    continue;
            }

        }
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator endkfB(itmp->first->kFMatchesB.end());
        for (kf = itmp->first->kFMatchesB.begin(); kf != endkfB; kf++)
        {
            if ( !kf->first->keyF )
                    continue;
            if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && (int)itmp->first->kFMatchesB.size() < minCount) )
                break;
            if ( !wrongMatches.empty() && std::find(wrongMatches.begin(), wrongMatches.end(), std::make_pair(kf->first, itmp->first)) != wrongMatches.end())
            {
                continue;
            }
            if ( itmp->first->GetIsOutlier() )
                break;
            KeyFrame* kftemp = kf->first;
            TrackedKeys& keys = kftemp->keysB;
            std::pair<int,int>& keyPos = kf->second;

            if ( kf->first->numb > lastActKF )
            {
                mpIsOut = false;
                continue;
            }
            timesIn ++;
            mpIsOut = false;
            ceres::CostFunction* costf;
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                const cv::KeyPoint& obs = keys.keyPoints[keyPos.first];
                close = keys.close[keyPos.first];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.keyPoints[keyPos.first].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(KB,tc1c2B, qc1c2B, obs2d, weight);
            }
            else if ( keyPos.second >= 0 )
            {
                const cv::KeyPoint& obs = keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.rightKeyPoints[keyPos.second].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(KB,tc1c2BR, qc1c2BR, obs2d, weight);
            }

            ordering->AddElementToGroup(itmp->second.data(), 0);
            if (localKFs.find(kf->first) != localKFs.end())
            {
                ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                if ( kf->first->fixed )
                {
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                    problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                }
            }
            else 
                continue;
            if ( close )
            {
                
                if ( keyPos.second < 0 )
                    continue;
                const cv::KeyPoint& obs = keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                const int oct {keys.rightKeyPoints[keyPos.second].octave};
                const double weight = (double)kftemp->InvSigmaFactor[oct];
                costf = LocalBundleAdjustmentR::Create(KB,tc1c2BR, qc1c2BR, obs2d, weight);
                if (localKFs.find(kf->first) != localKFs.end())
                {
                    ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
                    ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
                    problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
                    problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
                    if ( kf->first->fixed )
                    {
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
                        problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
                    }
                }
                else 
                    continue;
            }
        }
        if ( mpIsOut )
            mpOutliers[mpCount] = true;
    }
    
    ceres::Solver::Options options;
    options.linear_solver_ordering.reset(ordering);
    options.num_threads = 8;
    options.max_num_iterations = 50;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator allmp, allmpend(allMapPoints.end());
    for (allmp = allMapPoints.begin(); allmp != allmpend; allmp ++)
    {
        MapPoint* mp = allmp->first;
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(mp->kFMatches.end());
        for (kf = mp->kFMatches.begin(); kf != endkf; kf++)
        {
            KeyFrame* kfCand = kf->first;
            std::pair<int,int>& keyPos = kf->second;
            if ( localKFs.find(kfCand) == localKFs.end() )
                continue;
            cv::KeyPoint kp;
            bool right {false};
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                kp = kfCand->keys.keyPoints[keyPos.first];
                close = kfCand->keys.close[keyPos.first];
            }
            else if ( keyPos.second >= 0 )
            {
                kp = kfCand->keys.rightKeyPoints[keyPos.second];
                right = true;
            }
            Eigen::Vector2d obs( (double)kp.pt.x, (double)kp.pt.y);
            const int oct = kp.octave;
            const double weight = (double)kfCand->sigmaFactor[oct];
            Eigen::Vector3d tcw = localKFs[kfCand].block<3, 1>(0, 0);
            Eigen::Vector4d q_xyzw = localKFs[kfCand].block<4, 1>(3, 0);
            Eigen::Quaterniond qcw(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
            bool outlier {false};
            if ( right )
                outlier = checkOutlierR(K,qcr, tcr, obs, allmp->second, tcw, qcw, reprjThreshold * weight);
            else
                outlier = checkOutlier(K, obs, allmp->second, tcw, qcw, reprjThreshold * weight);

            if ( outlier )
                wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
            else
            {
                if ( close )
                {
                    if ( keyPos.second < 0 )
                        continue;
                    cv::KeyPoint kpR = kfCand->keys.rightKeyPoints[keyPos.second];
                    const int octR = kpR.octave;
                    const double weightR = (double)kfCand->sigmaFactor[octR];
                    Eigen::Vector2d obsr( (double)kpR.pt.x, (double)kpR.pt.y);
                    bool outlierR = checkOutlierR(K,qcr, tcr, obsr, allmp->second, tcw, qcw, reprjThreshold * weightR);
                    if ( outlierR )
                    {
                        wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
                    }
                }
            }

        }
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator endkfB(mp->kFMatchesB.end());
        for (kf = mp->kFMatchesB.begin(); kf != endkfB; kf++)
        {
            KeyFrame* kfCand = kf->first;
            std::pair<int,int>& keyPos = kf->second;
            if ( localKFs.find(kfCand) == localKFs.end() )
                continue;
            cv::KeyPoint kp;
            bool right {false};
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                kp = kfCand->keysB.keyPoints[keyPos.first];
                close = kfCand->keysB.close[keyPos.first];
            }
            else if ( keyPos.second >= 0 )
            {
                kp = kfCand->keysB.rightKeyPoints[keyPos.second];
                right = true;
            }
            Eigen::Vector2d obs( (double)kp.pt.x, (double)kp.pt.y);
            const int oct = kp.octave;
            const double weight = (double)kfCand->sigmaFactor[oct];
            Eigen::Vector3d tcw = localKFs[kfCand].block<3, 1>(0, 0);
            Eigen::Vector4d q_xyzw = localKFs[kfCand].block<4, 1>(3, 0);
            Eigen::Quaterniond qcw(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
            bool outlier {false};
            if ( right )
                outlier = checkOutlierR(KB,qc1c2BR, tc1c2BR, obs, allmp->second, tcw, qcw, reprjThreshold * weight);
            else
                outlier = checkOutlierR(KB,qc1c2B, tc1c2B, obs, allmp->second, tcw, qcw, reprjThreshold * weight);

            if ( outlier )
                wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
            else
            {
                if ( close )
                {
                    if ( keyPos.second < 0 )
                        continue;
                    cv::KeyPoint kpR = kfCand->keysB.rightKeyPoints[keyPos.second];
                    Eigen::Vector2d obsr( (double)kpR.pt.x, (double)kpR.pt.y);
                    const int octR = kpR.octave;
                    const double weightR = (double)kfCand->sigmaFactor[octR];
                    bool outlierR = checkOutlierR(KB,qc1c2BR, tc1c2BR, obsr, allmp->second, tcw, qcw, reprjThreshold * weightR);
                    if ( outlierR )
                    {
                        wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
                    }
                }
            }

        }
    }

    first = false;
    }
    std::lock_guard<std::mutex> lock(map->mapMutex);

    if ( !wrongMatches.empty() )
    {
        for (size_t wM {0}, endwM {wrongMatches.size()}; wM < endwM; wM ++)
        {
            KeyFrame* kF = wrongMatches[wM].first;
            MapPoint* mp = wrongMatches[wM].second;
            if ( mp->kFMatches.size() > 0 )
            {
                const std::pair<int,int>& keyPos = mp->kFMatches.at(kF);
                kF->eraseMPConnection(keyPos);
                mp->eraseKFConnection(kF);
            }
            if ( mp->kFMatchesB.size() > 0 )
            {
                const std::pair<int,int>& keyPos = mp->kFMatchesB.at(kF);
                kF->eraseMPConnectionB(keyPos);
                mp->eraseKFConnectionB(kF);
            }
        }
    }


    
    std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>::iterator localkf, endlocalkf(localKFs.end());
    for ( localkf = localKFs.begin(); localkf != endlocalkf; localkf++)
    {
        localkf->first->pose.setInvPose(Converter::Matrix_7_1_ToMatrix4d(localkf->second));
        localkf->first->setBackPose(localkf->first->pose.pose * zedPtr->TCamToCam);
        localkf->first->LBA = true;
    }

    int mpCount {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
    for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
    {
        if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && (int)itmp->first->kFMatches.size() < minCount && (int)itmp->first->kFMatchesB.size() < minCount) )
            itmp->first->SetIsOutlier(true);
        else
        {
            itmp->first->updatePos(itmp->second, zedPtr);
        }
    }

    

    map->endLCIdx = actKeyF.front()->numb;
    map->LCDone = true;
    map->LCStart = false;
    map->aprilTagDetected = false;
    std::cout << "Loop Closure Optimization Finished!" << std::endl;
}

void LocalMapper::beginLocalMapping()
{
    using namespace std::literals::chrono_literals;
    while ( !map->endOfFrames )
    {
        if ( map->keyFrameAdded && !map->LBADone && !map->LCStart )
        {

            std::vector<DC_VSLAM::KeyFrame *> actKeyF;
            KeyFrame* lastKF = map->keyFrames.at(map->kIdx - 1);
            actKeyF.reserve(20);
            actKeyF.emplace_back(lastKF);
            lastKF->getConnectedKFs(actKeyF, actvKFMaxSize);
            
            {
            triangulateNewPointsR(actKeyF);
            }
            {
            localBAR(actKeyF);
            }

        }
        if ( stopRequested )
            break;
        std::this_thread::sleep_for(20ms);
    }
    std::cout << "LocalMap Thread Exited!" << std::endl;
}

void LocalMapper::beginLocalMappingB()
{
    using namespace std::literals::chrono_literals;
    while ( !map->endOfFrames )
    {
        if ( map->keyFrameAdded && !map->LBADone && !map->LCStart )
        {
            std::vector<DC_VSLAM::KeyFrame *> actKeyF;
            KeyFrame* lastKF = map->keyFrames.at(map->kIdx - 1);
            actKeyF.reserve(20);
            actKeyF.emplace_back(lastKF);
            lastKF->getConnectedKFs(actKeyF, actvKFMaxSize);
            {
            std::thread front(&LocalMapper::triangulateNewPointsRB, this, std::ref(zedPtr), std::ref(actKeyF), false);
            std::thread back(&LocalMapper::triangulateNewPointsRB, this, std::ref(zedPtrB), std::ref(actKeyF), true);
            front.join();
            back.join();
            }
            {
            localBARB(actKeyF);
            }

        }
        if ( stopRequested )
            break;
        std::this_thread::sleep_for(20ms);
    }
    std::cout << "LocalMap Thread Exited!" << std::endl;

}

void LocalMapper::beginLoopClosure()
{
    using namespace std::literals::chrono_literals;
    while ( true )
    {
        if ( map->LCStart )
        {
            std::vector<DC_VSLAM::KeyFrame *> activeKF;
            activeKF.reserve(map->LCCandIdx);
            KeyFrame* kFLCCand = map->keyFrames.at(map->LCCandIdx);
            activeKF.emplace_back(kFLCCand);
            kFLCCand->getConnectedKFsLC(map, activeKF);

            loopClosureR(activeKF);

        }
        if ( stopRequested )
            break;
        std::this_thread::sleep_for(100ms);
    }
    std::cout << "LoopClosure Thread Exited!" << std::endl;
}

void LocalMapper::beginLoopClosureB()
{
    using namespace std::literals::chrono_literals;
    while ( true )
    {
        if ( map->LCStart )
        {
            std::vector<DC_VSLAM::KeyFrame *> activeKF;
            activeKF.reserve(map->LCCandIdx);
            KeyFrame* kFLCCand = map->keyFrames.at(map->LCCandIdx);
            activeKF.emplace_back(kFLCCand);
            kFLCCand->getConnectedKFsLC(map, activeKF);

            loopClosureRB(activeKF);

        }
        if ( stopRequested )
            break;
        std::this_thread::sleep_for(100ms);
    }
    std::cout << "LoopClosure Thread Exited!" << std::endl;
}

} // namespace DC_VSLAM