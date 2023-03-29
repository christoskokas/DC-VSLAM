#include "FeatureMatcher.h"

namespace DC_VSLAM
{

FeatureMatcher::FeatureMatcher(const Zed_Camera* _zed, const FeatureExtractor* _feLeft, const FeatureExtractor* _feRight, const int _imageHeight) : zedptr(_zed), feLeft(_feLeft), feRight(_feRight), imageHeight(_imageHeight)
{

}

void FeatureMatcher::getMatchIdxs(const cv::Point2f& predP, std::vector<int>& idxs, const TrackedKeys& keysLeft, const int predictedScale, const float radius, bool right)
{
    idxs.reserve(200);
    const float trackX = predP.x;
    const float trackY = predP.y;
    const int minX = std::max(0, cvFloor((trackX - radius)* keysLeft.xMult));
    const int maxX = std::min(keysLeft.xGrids - 1, cvCeil((trackX + radius) * keysLeft.xMult));
    const int minY = std::max(0, cvFloor((trackY - radius)* keysLeft.yMult));
    const int maxY = std::min(keysLeft.yGrids - 1, cvCeil((trackY + radius) * keysLeft.yMult));

    if ( minX >= keysLeft.xGrids )
        return;
    if ( minY >= keysLeft.yGrids )
        return;
    if ( maxX < 0 )
        return;
    if ( maxY < 0 )
        return;

    int offset {1};
    const int maxLevel {predictedScale + offset};
    const int minLevel {predictedScale - offset};


    for ( int row {minY}; row <= maxY; row ++)
    {
        for ( int col {minX}; col <= maxX; col++)
        {
            const std::vector<int>& grid = right ? keysLeft.rkeyGrid[row][col] : keysLeft.lkeyGrid[row][col];

            if ( grid.empty() )
                continue;

            for (size_t i {0}, end{grid.size()}; i < end; i++)
            {
                const cv::KeyPoint kpCand = right ? keysLeft.rightKeyPoints[grid[i]] : keysLeft.keyPoints[grid[i]];

                if (kpCand.octave > maxLevel || kpCand.octave < minLevel )
                    continue;

                const float distx = kpCand.pt.x-trackX;
                const float disty = kpCand.pt.y-trackY;

                if(fabs(distx)<radius && fabs(disty)<radius)
                    idxs.push_back(grid[i]);
                
            }

        }
    }
    
}

int FeatureMatcher::matchByProjectionRPredLBA(const KeyFrame* lastKF, KeyFrame* newKF, std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs, const float rad, const std::vector<std::pair<cv::Point2f, cv::Point2f>>& predPoints, const std::vector<float>& maxDistsScale, std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d)
{
    int nMatches {0};
    const TrackedKeys& lastKeys = lastKF->keys;
    const TrackedKeys& keysLeft = newKF->keys;
    const Eigen::Matrix4d toFindPose = newKF->pose.pose;
    for ( size_t i {0}, end{p4d.size()}; i < end; i++)
    {
        const Eigen::Vector4d& wPos = p4d[i].first;
        const std::pair<int,int>& keyPair = p4d[i].second;
        cv::Mat mpDesc;
        if ( keyPair.first >= 0 )
        {
            const MapPoint* mp = lastKF->localMapPoints[keyPair.first];
            if ( mp )
                mpDesc = mp->desc.clone();
            else
                mpDesc = lastKeys.Desc.row(keyPair.first).clone();
        }
        else
            mpDesc = lastKeys.rightDesc.row(keyPair.second).clone();
        
        Eigen::Vector3d pos = wPos.block<3,1>(0,0) - toFindPose.block<3,1>(0,3);
        const float dist = pos.norm();
        float dif = maxDistsScale[i]/dist;
        int predScale = cvCeil(log(dif)/lastKF->logScale);
        if ( predScale < 0 )
            predScale = 0;
        else if ( predScale >= lastKF->nScaleLev )
            predScale = lastKF->nScaleLev - 1;

        int bestDist = 256;
        int bestIdx = -1;
        int bestLev = -1;
        int bestLev2 = -1;
        int secDist = 256;
        const cv::Point2f& pLeft = predPoints[i].first;
        float radius = feLeft->scalePyramid[predScale] * rad;
        if ( pLeft.x > 0 && pLeft.y > 0 )
        {
            cv::Point2f prevKeyPos;
            if ( keyPair.first >= 0 )
                prevKeyPos = lastKeys.keyPoints[keyPair.first].pt;
            else
                prevKeyPos = lastKeys.rightKeyPoints[keyPair.second].pt;
            std::vector<int> idxs;
            getMatchIdxs(pLeft, idxs, keysLeft, predScale, radius, false);

            if ( !idxs.empty() )
            {
                for (auto& idx : idxs)
                {
                    if ( newKF->unMatchedF[idx] >= 0 )
                        continue;
                    const cv::KeyPoint& kPL = keysLeft.keyPoints[idx];
                    const int kpllevel = kPL.octave;
                    int dist = DescriptorDistance(mpDesc, keysLeft.Desc.row(idx));
                    if ( dist < bestDist)
                    {
                        secDist = bestDist;
                        bestLev2 = bestLev;
                        bestDist = dist;
                        bestLev = kpllevel;
                        bestIdx = idx;
                        continue;
                    }
                    if ( dist < secDist)
                    {
                        secDist = dist;
                        bestLev2 = bestLev;
                    }
                }
            }

        }


        radius = feLeft->scalePyramid[predScale] * rad;
        
        const cv::Point2f& pRight = predPoints[i].second;

        int bestDistR = 256;
        int bestIdxR = -1;
        int bestLevR = -1;
        int bestLevR2 = -1;
        int secDistR = 256;

        if ( pRight.x > 0 && pRight.y > 0 )
        {
            cv::Point2f prevKeyPos;
            if ( keyPair.second >= 0 )
                prevKeyPos = lastKeys.rightKeyPoints[keyPair.second].pt;
            else
                prevKeyPos = lastKeys.keyPoints[keyPair.first].pt;
            std::vector<int> idxs;
            getMatchIdxs(pRight, idxs, keysLeft, predScale, radius, true);

            if ( !idxs.empty() )
            {
                for (auto& idx : idxs)
                {
                    if ( newKF->unMatchedFR[idx] >= 0 )
                        continue;
                    const cv::KeyPoint& kPL = keysLeft.rightKeyPoints[idx];
                    const int kpllevel = kPL.octave;
                    int dist = DescriptorDistance(mpDesc, keysLeft.rightDesc.row(idx));
                    if ( dist < bestDistR)
                    {
                        secDistR = bestDistR;
                        bestLevR2 = bestLevR;
                        bestDistR = dist;
                        bestLevR = kpllevel;
                        bestIdxR = idx;
                        continue;
                    }
                    if ( dist < secDistR)
                    {
                        secDistR = dist;
                        bestLevR2 = kpllevel;
                    }
                }
            }

        }

        bool right = false;
        if ( bestDist > bestDistR)
        {
            bestDist = bestDistR;
            secDist = secDistR;
            bestLev = bestLevR;
            bestLev2 = bestLevR2;
            right = true;
        }

        if ( bestDist > matchDistLBA )
            continue;
        
        if (bestLev == bestLev2 && bestDist >= ratioLBA * secDist )
            continue;
        if (bestLev != bestLev2 || bestDist < ratioLBA * secDist)
        {
            nMatches ++;
            if ( right )
            {
                int rIdx {bestIdxR}, lIdx {-1};
                if ( keysLeft.leftIdxs[bestIdxR] >= 0 )
                {
                    lIdx = keysLeft.leftIdxs[bestIdxR];
                }
                matchedIdxs[i].emplace_back(std::make_pair(newKF, std::make_pair(lIdx,rIdx)));

            }
            else
            {
                int rIdx {-1}, lIdx {bestIdx};
                if ( keysLeft.rightIdxs[bestIdx] >= 0 )
                {
                    rIdx = keysLeft.rightIdxs[bestIdx];
                }
                matchedIdxs[i].emplace_back(std::make_pair(newKF, std::make_pair(lIdx,rIdx)));

            }
        }
    }
    return nMatches;
}

int FeatureMatcher::matchByProjectionRPredLBAB(const Zed_Camera* zedCam, const KeyFrame* lastKF, KeyFrame* newKF, std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs, const float rad, const std::vector<std::pair<cv::Point2f, cv::Point2f>>& predPoints, const std::vector<float>& maxDistsScale, std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, const bool back)
{
    int nMatches {0};
    const Eigen::Matrix4d& toFindPose = (back) ? newKF->backPose : newKF->pose.pose;
    const std::vector<MapPoint*>& localMps = (back) ? lastKF->localMapPointsB : lastKF->localMapPoints;
    const std::vector<int>& unMatchVec = (back) ? newKF->unMatchedFB : newKF->unMatchedF;
    const std::vector<int>& unMatchVecR = (back) ? newKF->unMatchedFRB : newKF->unMatchedFR;
    const TrackedKeys& lastKeys = (back) ? lastKF->keysB : lastKF->keys;
    const TrackedKeys& keysLeft = (back) ? newKF->keysB : newKF->keys;
    for ( size_t i {0}, end{p4d.size()}; i < end; i++)
    {
        const Eigen::Vector4d& wPos = p4d[i].first;
        const std::pair<int,int>& keyPair = p4d[i].second;
        cv::Mat mpDesc;
        if ( keyPair.first >= 0 )
        {
            const MapPoint* mp = localMps[keyPair.first];
            if ( mp )
                mpDesc = mp->desc.clone();
            else
                mpDesc = lastKeys.Desc.row(keyPair.first).clone();
        }
        else
            mpDesc = lastKeys.rightDesc.row(keyPair.second).clone();
        
        Eigen::Vector3d pos = wPos.block<3,1>(0,0) - toFindPose.block<3,1>(0,3);
        const float dist = pos.norm();
        float dif = maxDistsScale[i]/dist;
        int predScale = cvCeil(log(dif)/lastKF->logScale);
        if ( predScale < 0 )
            predScale = 0;
        else if ( predScale >= lastKF->nScaleLev )
            predScale = lastKF->nScaleLev - 1;

        int bestDist = 256;
        int bestIdx = -1;
        int bestLev = -1;
        int bestLev2 = -1;
        int secDist = 256;
        const cv::Point2f& pLeft = predPoints[i].first;
        float radius = feLeft->scalePyramid[predScale] * rad;
        if ( pLeft.x > 0 && pLeft.y > 0 )
        {
            cv::Point2f prevKeyPos;
            if ( keyPair.first >= 0 )
                prevKeyPos = lastKeys.keyPoints[keyPair.first].pt;
            else
                prevKeyPos = lastKeys.rightKeyPoints[keyPair.second].pt;
            std::vector<int> idxs;
            getMatchIdxs(pLeft, idxs, keysLeft, predScale, radius, false);

            if ( !idxs.empty() )
            {
                for (auto& idx : idxs)
                {
                    if ( unMatchVec[idx] >= 0 )
                        continue;
                    const cv::KeyPoint& kPL = keysLeft.keyPoints[idx];
                    const int kpllevel = kPL.octave;
                    int dist = DescriptorDistance(mpDesc, keysLeft.Desc.row(idx));
                    if ( dist < bestDist)
                    {
                        secDist = bestDist;
                        bestLev2 = bestLev;
                        bestDist = dist;
                        bestLev = kpllevel;
                        bestIdx = idx;
                        continue;
                    }
                    if ( dist < secDist)
                    {
                        secDist = dist;
                        bestLev2 = bestLev;
                    }
                }
            }

        }


        radius = feLeft->scalePyramid[predScale] * rad;
        
        const cv::Point2f& pRight = predPoints[i].second;

        int bestDistR = 256;
        int bestIdxR = -1;
        int bestLevR = -1;
        int bestLevR2 = -1;
        int secDistR = 256;

        if ( pRight.x > 0 && pRight.y > 0 )
        {
            cv::Point2f prevKeyPos;
            if ( keyPair.second >= 0 )
                prevKeyPos = lastKeys.rightKeyPoints[keyPair.second].pt;
            else
                prevKeyPos = lastKeys.keyPoints[keyPair.first].pt;
            std::vector<int> idxs;
            getMatchIdxs(pRight, idxs, keysLeft, predScale, radius, true);

            if ( !idxs.empty() )
            {
                for (auto& idx : idxs)
                {
                    if ( unMatchVecR[idx] >= 0 )
                        continue;
                    const cv::KeyPoint& kPL = keysLeft.rightKeyPoints[idx];
                    const int kpllevel = kPL.octave;
                    int dist = DescriptorDistance(mpDesc, keysLeft.rightDesc.row(idx));
                    if ( dist < bestDistR)
                    {
                        secDistR = bestDistR;
                        bestLevR2 = bestLevR;
                        bestDistR = dist;
                        bestLevR = kpllevel;
                        bestIdxR = idx;
                        continue;
                    }
                    if ( dist < secDistR)
                    {
                        secDistR = dist;
                        bestLevR2 = kpllevel;
                    }
                }
            }

        }

        bool right = false;
        if ( bestDist > bestDistR)
        {
            bestDist = bestDistR;
            secDist = secDistR;
            bestLev = bestLevR;
            bestLev2 = bestLevR2;
            right = true;
        }

        if ( bestDist > matchDistLBA )
            continue;
        
        if (bestLev == bestLev2 && bestDist >= ratioLBA * secDist )
            continue;
        if (bestLev != bestLev2 || bestDist < ratioLBA * secDist)
        {
            nMatches ++;
            if ( right )
            {
                int rIdx {bestIdxR}, lIdx {-1};
                if ( keysLeft.leftIdxs[bestIdxR] >= 0 )
                {
                    lIdx = keysLeft.leftIdxs[bestIdxR];
                }
                matchedIdxs[i].emplace_back(std::make_pair(newKF, std::make_pair(lIdx,rIdx)));

            }
            else
            {
                int rIdx {-1}, lIdx {bestIdx};
                if ( keysLeft.rightIdxs[bestIdx] >= 0 )
                {
                    rIdx = keysLeft.rightIdxs[bestIdx];
                }
                matchedIdxs[i].emplace_back(std::make_pair(newKF, std::make_pair(lIdx,rIdx)));

            }
        }
    }
    return nMatches;
}

int FeatureMatcher::matchByProjectionRPred(std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<int>& matchedIdxsL, std::vector<int>& matchedIdxsR, std::vector<std::pair<int,int>>& matchesIdxs, const float rad)
{
    int nMatches {0};

    for ( size_t i {0}, prevE{activeMapPoints.size()}; i < prevE; i++)
    {
        std::pair<int,int>& keyPair = matchesIdxs[i];
        MapPoint* mp = activeMapPoints[i];
        if ( keyPair.first >= 0 || keyPair.second >= 0 )
            continue;
        const cv::Mat& mpDesc = mp->desc;
        int predScaleLevel = mp->scaleLevelL;

        cv::Point2f& pLeft = mp->predL;

        float radius = feLeft->scalePyramid[predScaleLevel] * rad;

        std::vector<int> idxs;
        getMatchIdxs(pLeft, idxs, keysLeft, predScaleLevel, radius, false);

        int bestDist = 256;
        int bestIdx = -1;
        int bestLev = -1;
        int bestLev2 = -1;
        int secDist = 256;
        if ( !idxs.empty() && mp->inFrame )
        {
            for (auto& idx : idxs)
            {
                if ( matchedIdxsL[idx] >= 0 )
                    continue;
                cv::KeyPoint& kPL = keysLeft.keyPoints[idx];
                const int kpllevel = kPL.octave;
                int dist = DescriptorDistance(mpDesc, keysLeft.Desc.row(idx));
                if ( dist < bestDist)
                {
                    secDist = bestDist;
                    bestLev2 = bestLev;
                    bestDist = dist;
                    bestLev = kpllevel;
                    bestIdx = idx;
                    continue;
                }
                if ( dist < secDist)
                {
                    secDist = dist;
                    bestLev2 = kpllevel;
                }
            }
        }


        std::vector<int> idxsR;
        predScaleLevel = mp->scaleLevelR;
        radius = feLeft->scalePyramid[predScaleLevel] * rad;
        
        cv::Point2f& pRight = mp->predR;
        getMatchIdxs(pRight, idxsR, keysLeft, predScaleLevel, radius, true);

        int bestDistR = 256;
        int bestIdxR = -1;
        int bestLevR = -1;
        int bestLevR2 = -1;
        int secDistR = 256;

        if ( !idxsR.empty() && mp->inFrameR )
        {
            for (auto& idx : idxsR)
            {
                if ( matchedIdxsR[idx] >= 0 )
                    continue;
                cv::KeyPoint& kPL = keysLeft.rightKeyPoints[idx];
                const int kpllevel = kPL.octave;
                int dist = DescriptorDistance(mpDesc, keysLeft.rightDesc.row(idx));
                if ( dist < bestDistR)
                {
                    secDistR = bestDistR;
                    bestLevR2 = bestLevR;
                    bestDistR = dist;
                    bestLevR = kpllevel;
                    bestIdxR = idx;
                    continue;
                }
                if ( dist < secDistR)
                {
                    secDistR = dist;
                    bestLevR2 = kpllevel;
                }
            }
        }

        bool right = false;
        if ( bestDist > bestDistR)
        {
            bestDist = bestDistR;
            secDist = secDistR;
            bestLev = bestLevR;
            bestLev2 = bestLevR2;
            right = true;
        }

        if ( bestDist > matchDistProj )
            continue;
        
        if ( bestLev == bestLev2 && bestDist >= ratioProj * secDist )
            continue;
        if ( bestLev != bestLev2 || bestDist < ratioProj * secDist )
        {
            nMatches ++;
            if ( right )
            {
                matchedIdxsR[bestIdxR] = i;
                keyPair.second = bestIdxR;
                if ( keysLeft.leftIdxs[bestIdxR] >= 0 )
                {
                    keyPair.first = keysLeft.leftIdxs[bestIdxR];
                    matchedIdxsL[keysLeft.leftIdxs[bestIdxR]] = i;

                }

            }
            else
            {
                matchedIdxsL[bestIdx] = i;
                keyPair.first = bestIdx;
                if ( keysLeft.rightIdxs[bestIdx] >= 0 )
                {
                    keyPair.second = keysLeft.rightIdxs[bestIdx];
                    matchedIdxsR[keysLeft.rightIdxs[bestIdx]] = i;

                }
            }
        }
    }
    return nMatches;
}

void FeatureMatcher::findStereoMatchesORB2R(const cv::Mat& lImage, const cv::Mat& rImage, const cv::Mat& rightDesc,  std::vector<cv::KeyPoint>& rightKeys, TrackedKeys& keysLeft)
{
    std::vector<std::vector < int > > indexes;
    destributeRightKeys(rightKeys, indexes);

    const size_t leftEnd {keysLeft.keyPoints.size()};

    keysLeft.estimatedDepth.resize(leftEnd, -1.0f);
    keysLeft.close.resize(leftEnd, false);
    keysLeft.rightIdxs.resize(leftEnd, -1);
    keysLeft.leftIdxs.resize(rightKeys.size(), -1);

    int leftRow {0};
    int matchesCount {0};

    const float minD = 0;
    const float maxD = zedptr->cameraLeft.fx;

    const int windowRadius {5};

    const int windowMovementX {5};
    std::vector<std::pair<float,int>> allDepths;
    allDepths.reserve(keysLeft.keyPoints.size());
    std::vector<std::pair<int,int>> allDists2;
    allDists2.reserve(keysLeft.keyPoints.size());
    std::vector < cv::KeyPoint >::const_iterator it,end(keysLeft.keyPoints.end());
    for (it = keysLeft.keyPoints.begin(); it != end; it++, leftRow++)
    {
        const int yKey = cvRound(it->pt.y);
        const float uL {it->pt.y};
        const int octL {it->octave};


        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = 256;
        int bestIdx = -1;



        const size_t endCount {indexes[yKey].size()};
        if (endCount == 0)
            continue;
        for (size_t allIdx {0};allIdx < endCount; allIdx++)
        {
            const int idx {indexes[yKey][allIdx]};
            const float uR {rightKeys[idx].pt.y};
            const int octR {rightKeys[idx].octave};

            if(octR < octL-1 || octR > octL+1)
                continue;
            if(!(uR>=minU && uR<=maxU))
                continue;

            int dist {DescriptorDistance(keysLeft.Desc.row(leftRow),rightDesc.row(idx))};

            if (bestDist > dist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }
        if (bestDist > thDist)
            continue;

        cv::KeyPoint& kR = rightKeys[bestIdx];

        const float kRx = kR.pt.x;
        const float scale = feLeft->scaleInvPyramid[octL];
        const float scuL = round(it->pt.x*scale);
        const float scvL = round(it->pt.y*scale);
        const float scuR = round(kRx*scale);


        const cv::Mat lWin = feLeft->imagePyramid[octL].rowRange(scvL - windowRadius, scvL + windowRadius + 1).colRange(scuL - windowRadius, scuL + windowRadius + 1);

        int bestDistW {INT_MAX};
        int bestX {0};

        std::vector < float > allDists;
        allDists.resize(2*windowMovementX + 1);




        for (int32_t xMov {-windowMovementX}; xMov <= windowMovementX ; xMov++)
        {
            const float startW = scuR + xMov - windowRadius;
            const float endW = scuR + xMov + windowRadius + 1;
            if ( startW < 0 || endW >= feRight->imagePyramid[octL].cols)
                continue;
            const cv::Mat rWin = feRight->imagePyramid[octL].rowRange(scvL - windowRadius, scvL + windowRadius + 1).colRange(startW, endW);

            float dist = cv::norm(lWin,rWin, cv::NORM_L1);
            if (bestDistW > dist)
            {
                bestX = xMov;
                bestDistW = dist;
            }
            allDists[xMov + windowMovementX] = dist;
        }
        if ((bestX == -windowMovementX) || (bestX == windowMovementX))
            continue;

        const float dist1 = allDists[windowMovementX + bestX-1];
        const float dist2 = allDists[windowMovementX + bestX];
        const float dist3 = allDists[windowMovementX + bestX+1];

        const float delta = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

        if (delta > 1 || delta < -1)
            continue;

        matchesCount++;
        

        const float newuR = feLeft->scalePyramid[octL] * ((float)scuR + (float)bestX + delta);


        const float disparity {it->pt.x - newuR};
        if (disparity > 0.0f && disparity < zedptr->cameraLeft.fx)
        {
            const float depth {((float)zedptr->cameraLeft.fx * zedptr->mBaseline)/disparity};
            keysLeft.rightIdxs[leftRow] = bestIdx;
            keysLeft.leftIdxs[bestIdx] = leftRow;
            keysLeft.estimatedDepth[leftRow] = depth;
            allDists2.emplace_back(bestDistW,leftRow);
            allDepths.emplace_back(depth,leftRow);
            // if false depth is unusable
            if (depth < zedptr->mBaseline * closeNumber)
            {
                keysLeft.close[leftRow] = true;

            }
        }

    }


    if ( allDepths.size() <= 0 )
        return;

    std::sort(allDepths.begin(), allDepths.end());
    std::sort(allDists2.begin(), allDists2.end());
    const int medianD {allDists2[allDists2.size()/2].first};
    const float medDistD = (float)medianD*(1.5f*1.4f);
    
    const int endDe {cvFloor(allDepths.size()*0.01)};
    for(int i=0;i < endDe;i++)
    {
        const int rIdx = keysLeft.rightIdxs[allDepths[i].second];
        if ( rIdx >= 0)
            keysLeft.leftIdxs[rIdx] = -1;
        keysLeft.rightIdxs[allDepths[i].second] = -1;
        keysLeft.estimatedDepth[allDepths[i].second] = -1;
        keysLeft.close[allDepths[i].second] = false;
    }

    

    for(int i=allDists2.size()-1;i>=0;i--)
    {
        if(allDists2[i].first<medDistD)
            break;
        else
        {
            const int rIdx = keysLeft.rightIdxs[allDists2[i].second];
            if ( rIdx >= 0)
                keysLeft.leftIdxs[rIdx] = -1;
            keysLeft.rightIdxs[allDists2[i].second] = -1;
            keysLeft.estimatedDepth[allDists2[i].second] = -1;
            keysLeft.close[allDists2[i].second] = false;
        }
    }


}

int FeatureMatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

void FeatureMatcher::destributeRightKeys(const std::vector < cv::KeyPoint >& rightKeys, std::vector<std::vector < int > >& indexes)
{


    indexes.resize(imageHeight);

    for (int32_t i = 0; i < imageHeight; i++)
        indexes[i].reserve(200);

    std::vector<cv::KeyPoint>::const_iterator it,end(rightKeys.end());
    int count {0};
    for (it = rightKeys.begin(); it != end; it++, count ++)
    {
        const int yKey = cvRound((*it).pt.y);
        const float r = 2.0f*feRight->scalePyramid[it->octave];
        const int mn = cvFloor(yKey - r);
        const int mx = cvCeil(yKey + r);

        

        for (int32_t pos = mn; pos <= mx; pos++)
            indexes[pos].emplace_back(count);
    }

}

} // namespace DC_VSLAM