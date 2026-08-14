#include <mapping.h>


namespace SlamDemo {
    using namespace std::chrono_literals;
    using namespace std;

    StereoBlockMatch matchSingleBlock(
        const double minVariance,
        const double minCorrelation,
        const cv::Size resolution,
        const int halfBlockSize,
        const int searchBound,
        const int centerX,
        const int centerY,
        const cv::Mat& combinedTSLeft,
        const cv::Mat& combinedTSRight
    ) {
        int blockSize = 2*halfBlockSize + 1;
        int blockArea = blockSize*blockSize;
        int leftX = centerX - halfBlockSize;
        int topY = centerY - halfBlockSize;
        cv::Rect leftRect(leftX, topY, blockSize, blockSize);
        //cout << "Left rectangle: " << leftRect << endl;

        cv::Mat leftPatch = combinedTSLeft(leftRect);
        cv::Scalar leftMean = cv::mean(leftPatch);
        cv::Mat znPatch = leftPatch - leftMean;
        cv::Scalar leftVariance = cv::sum(znPatch.mul(znPatch))/((double)blockArea - 1.);

        if(leftVariance[0] < minVariance) {
            return {centerX, centerY, -1, 0.};
        }

        int numSearches = min(searchBound, 1 + leftX);

        cv::Rect rightRect = leftRect;
        cv::Mat rightPatch = combinedTSRight(rightRect);
        cv::Scalar sumIntensity = cv::sum(rightPatch);
        cv::Scalar sumSquaredIntensity = cv::sum(rightPatch.mul(rightPatch));
        cv::Scalar variance = (sumSquaredIntensity - sumIntensity*sumIntensity)/((double)blockArea - 1.);
        cv::Scalar covariance = (cv::sum(leftPatch.mul(rightPatch)) - leftMean*sumIntensity)/((double)blockArea - 1.);
        cv::Mat lastPatch = rightPatch;

        double bestCorrelation = minCorrelation;
        int pixelDisparity = -1;

        if(variance[0] > minVariance) {
            double correlation = covariance[0]/sqrt(leftVariance[0]*variance[0]);
            if(correlation > bestCorrelation) {
                bestCorrelation = correlation;
                pixelDisparity = 0;
            }
        }

        for(int disparity = 1; disparity < numSearches; disparity++) {
            rightRect = cv::Rect(leftX - disparity, topY, blockSize, blockSize);
            rightPatch = combinedTSRight(rightRect);
            cv::Mat lastCol = lastPatch.col(blockSize - 1);
            sumIntensity -= cv::sum(lastCol);
            cv::Mat nextCol = rightPatch.col(0);
            sumIntensity += cv::sum(nextCol);
            sumSquaredIntensity -= cv::sum(lastCol.mul(lastCol));
            sumSquaredIntensity += cv::sum(nextCol.mul(nextCol));

            lastPatch = rightPatch;

            variance = (sumSquaredIntensity - sumIntensity*sumIntensity)/((double)blockArea);
            covariance = (cv::sum(leftPatch.mul(rightPatch)) - leftMean*sumIntensity)/((double)blockArea);

            if(variance[0] > minVariance) {
                double correlation = covariance[0]/sqrt(leftVariance[0]*variance[0]);
                if(correlation > minCorrelation) {
                    bestCorrelation = correlation;
                    pixelDisparity = disparity;
                }
            }
        }

        return {centerX, centerY, pixelDisparity, bestCorrelation};
    }

    void singleBlockSearch(
        const double minVariance,
        const double minCorrelation,
        cv::Scalar& leftVariance,
        vector<cv::Scalar>& rightVarianceAtDisparity,
        vector<cv::Scalar>& covarianceAtDisparity,
        int& match,
        double& bestCorrelation
    ) {
        // -1 for no match
        match = -1;
        bestCorrelation = 0.;
        if(leftVariance[0] < minVariance) {
            return;
        }

        for(int disparity = 0; disparity < covarianceAtDisparity.size(); disparity++) {
            if(rightVarianceAtDisparity[disparity][0] > minVariance) {
                double corr = covarianceAtDisparity[disparity][0]/(sqrt(leftVariance[0]*rightVarianceAtDisparity[disparity][0]));
                if(corr > minCorrelation && corr > bestCorrelation) {
                    match = disparity;
                    corr = bestCorrelation;
                }
            }
        }
    }

    vector<StereoBlockMatch> stereoBlockMatchingSequential(
        const double minVariance,
        const double minCorrelation,
        const cv::Size resolution,
        const int halfBlockSize,
        const int searchBound,
        const cv::Mat& combinedTSLeft,
        const cv::Mat& combinedTSRight,
        const vector<int>& xCenters,
        const vector<int>& yCenters,
        const int start,
        const int end
    ) {
        int numBlocks = end - start;
        vector<StereoBlockMatch> matches;
        matches.reserve(numBlocks);

        for(int i = start; i < end; i++) {
            matches.push_back(matchSingleBlock(
                minVariance,
                minCorrelation,
                resolution,
                halfBlockSize,
                searchBound,
                xCenters[i],
                yCenters[i],
                combinedTSLeft,
                combinedTSRight
            ));
        }

        return matches;
    }

    vector<StereoBlockMatch> stereoBlockMatchingParallel(
        const int numThreads,
        const double minVariance,
        const double minCorrelation,
        const cv::Size resolution,
        const int halfBlockSize,
        const int searchBound,
        const cv::Mat& combinedTSLeft,
        const cv::Mat& combinedTSRight,
        const vector<int>& xCenters,
        const vector<int>& yCenters
    ) {
        size_t numBlocks = xCenters.size();
        vector<StereoBlockMatch> matches;
        matches.reserve(numBlocks);
        vector<future<vector<StereoBlockMatch>>> futures;
        futures.reserve(numThreads);

        int blocksPerThread = numBlocks/numThreads;

        for(int i = 0; i < numThreads; i++) {
            int start = blocksPerThread*i;
            int end = min((int)numBlocks, blocksPerThread*(i + 1));
            futures.push_back(async(launch::async, [&] {
                return stereoBlockMatchingSequential(
                    minVariance,
                    minCorrelation,
                    resolution,
                    halfBlockSize,
                    searchBound,
                    combinedTSLeft,
                    combinedTSRight,
                    xCenters,
                    yCenters,
                    start,
                    end);
                }
            ));
        }

        for(int i = 0; i < numThreads; i++) {
            auto theseMatches = futures[i].get();
            matches.insert(matches.end(), theseMatches.begin(), theseMatches.end());
        }

        return matches;
    }

    cv::Mat drawBlockMatchingResult(
        const cv::Size resolution,
        const int searchBound,
        const vector<StereoBlockMatch>& sbmResult
    ){

        // render horizontal and vertical depth estimations separately
        // use hue to indicate disparity, value to indicate confidence
        cv::Mat visHSV = cv::Mat::zeros(resolution, CV_8UC3);
        cv::Mat visBGR;
        for(auto match : sbmResult) {
            if(match.pixelDisparity > -1) {
                uint8_t hue = (uint8_t)(120.*match.pixelDisparity/searchBound);
                uint8_t val = 255;//(uint8_t)(255.*sbmResult.correlation[i]);
                for(int i = match.x - 1; i < match.x + 2; i++) {
                    for(int j = match.y - 1; j < match.y + 2; j++) {
                        visHSV.at<cv::Vec3b>(j, i) = cv::Vec3b(hue, 255, val);
                    }
                }
            }
        }
        cv::cvtColor(visHSV, visBGR, cv::COLOR_HSV2BGR);

        return visBGR;
    }

    tuple<double, double> sobelAtPoint(cv::Mat img, int y, int x)
    {
        double dx = (img.at<double>(x + 1, y) + 0.5 * (img.at<double>(x + 1, y + 1) + img.at<double>(x + 1, y + 1))) - (img.at<double>(x - 1, y) + 0.5 * (img.at<double>(x - 1, y + 1) + img.at<double>(x - 1, y + 1)));
        double dy = (img.at<double>(x, y + 1) + 0.5 * (img.at<double>(x + 1, y + 1) + img.at<double>(x - 1, y + 1))) - (img.at<double>(x, y - 1) + 0.5 * (img.at<double>(x + 1, y - 1) + img.at<double>(x - 1, y - 1)));
        return {dx, dy};
    }

    void depthEstimationLoop(
        const int numThreads,
        const double minVariance,
        const double minCorrelation,
        const cv::Size resolution,
        const int halfBlockSize,
        const int maxNumBlocks,
        const int searchBound,
        queue<dv::EventStore>& incomingLeftEvents,
        queue<cv::Mat>& incomingLeftImages,
        queue<cv::Mat>& incomingRightImages,
        queue<vector<cv::Mat>>& outgoingImages
    ) {
        while(true) {
            if(!incomingLeftImages.empty() && !incomingRightImages.empty() && !incomingLeftEvents.empty()) {
                dv::EventStore events = incomingLeftEvents.front();
                incomingLeftEvents.pop();
                cv::Mat leftImage = incomingLeftImages.front();
                incomingLeftImages.pop();
                cv::Mat rightImage = incomingRightImages.front();
                incomingRightImages.pop();

                vector<int> xCentersVertical, yCentersVertical, xCentersHorizontal, yCentersHorizontal;
                xCentersVertical.reserve(maxNumBlocks);
                yCentersVertical.reserve(maxNumBlocks);
                xCentersVertical.reserve(maxNumBlocks);
                yCentersVertical.reserve(maxNumBlocks);
                for(int i = max(0, (int)events.size() - maxNumBlocks); i < events.size(); i++) {
                    auto ev = events[i];
                    if(ev.x() >= halfBlockSize && ev.x() < resolution.width - halfBlockSize && ev.y() >= halfBlockSize && ev.y() < resolution.height - halfBlockSize) {
                        auto sobel = sobelAtPoint(leftImage, ev.x(), ev.y());
                        if(get<0>(sobel) > get<1>(sobel)) {
                            xCentersVertical.push_back(ev.x());
                            yCentersVertical.push_back(ev.y());
                        }
                        else {
                            xCentersHorizontal.push_back(ev.x());
                            yCentersHorizontal.push_back(ev.y());
                        }
                    }
                }

                vector<StereoBlockMatch> verticalMatchResult = stereoBlockMatchingParallel(
                    numThreads,
                    minVariance,
                    minCorrelation,
                    resolution,
                    halfBlockSize,
                    searchBound,
                    ref(leftImage),
                    ref(rightImage),
                    ref(xCentersVertical),
                    ref(yCentersVertical)
                );
                vector<StereoBlockMatch> horizontalMatchResult = stereoBlockMatchingParallel(
                    numThreads,
                    minVariance,
                    minCorrelation,
                    resolution,
                    halfBlockSize,
                    searchBound,
                    ref(leftImage),
                    ref(rightImage),
                    ref(xCentersHorizontal),
                    ref(yCentersHorizontal)
                );

                // render events based on horizontal or vertical orientation
                cv::Mat eventVis = cv::Mat::zeros(resolution, CV_8UC3);
                for(int i = 0; i < xCentersVertical.size(); i++) {
                    int x = xCentersVertical[i];
                    int y = yCentersVertical[i];
                    for(int i = max(0, x - 1); i < min(resolution.width, x + 2); i++) {
                        for(int j = max(0, y - 1); j < min(resolution.height, y + 2); j++) {
                            eventVis.at<cv::Vec3b>(j, i) = cv::Vec3b(255, 0, 0);
                        }
                    }
                }
                for(int i = 0; i < xCentersHorizontal.size(); i++) {
                    int x = xCentersHorizontal[i];
                    int y = yCentersHorizontal[i];
                    for(int i = max(0, x - 1); i < min(resolution.width, x + 2); i++) {
                        for(int j = max(0, y - 1); j < min(resolution.height, y + 2); j++) {
                            eventVis.at<cv::Vec3b>(j, i) = cv::Vec3b(0, 0, 255);
                        }
                    }
                }

                cv::Mat verticalVis = drawBlockMatchingResult(resolution, searchBound, verticalMatchResult);
                cv::Mat horizontalVis = drawBlockMatchingResult(resolution, searchBound, horizontalMatchResult);

                vector<cv::Mat> outgoing = {eventVis, verticalVis, horizontalVis};

                outgoingImages.push(outgoing);
            }
            else {
                this_thread::sleep_for(2ms);
            }
        }
    }
}