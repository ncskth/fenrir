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
        const int blockSize = 2*halfBlockSize + 1;
        const int blockArea = blockSize*blockSize;
        const int leftX = centerX - halfBlockSize;
        const int topY = centerY - halfBlockSize;

        // Bounds check
        if (leftX < 0 || topY < 0 ||
            leftX + blockSize > combinedTSLeft.cols ||
            topY + blockSize > combinedTSLeft.rows) {
            return {centerX, centerY, -1, 0.};
        }

        // Access left patch directly from matrix data
        const double* leftData = combinedTSLeft.ptr<double>(topY) + leftX;
        const size_t leftStep = combinedTSLeft.step / sizeof(double);

        // Compute left mean and variance directly
        double leftSum = 0, leftSumSq = 0;
        for(int y = 0; y < blockSize; y++) {
            const double* row = leftData + y * leftStep;
            for(int x = 0; x < blockSize; x++) {
                const double val = row[x];
                leftSum += val;
                leftSumSq += val * val;
            }
        }
        const double leftMean = leftSum / blockArea;
        const double leftVariance = (leftSumSq - leftSum * leftSum / blockArea) / (blockArea - 1);

        if(leftVariance < minVariance) {
            return {centerX, centerY, -1, 0.};
        }

        const int maxDisparity = std::min(searchBound, leftX + 1);
        if(maxDisparity < 0) {
            return {centerX, centerY, -1, 0.};
        }

        // Initial right patch at disparity 0
        const double* rightData = combinedTSRight.ptr<double>(topY) + leftX;
        const size_t rightStep = combinedTSRight.step / sizeof(double);

        // Compute initial statistics
        double rightSum = 0, rightSumSq = 0, sumXY = 0;
        for(int y = 0; y < blockSize; y++) {
            const double* leftRow = leftData + y * leftStep;
            const double* rightRow = rightData + y * rightStep;
            for(int x = 0; x < blockSize; x++) {
                const double lv = leftRow[x];
                const double rv = rightRow[x];
                rightSum += rv;
                rightSumSq += rv * rv;
                sumXY += lv * rv;
            }
        }

        double variance = (rightSumSq - rightSum * rightSum / blockArea) / (blockArea - 1);
        double covariance = (sumXY - leftSum * rightSum / blockArea) / (blockArea - 1);

        double bestCorrelation = minCorrelation;
        int pixelDisparity = -1;

        if(variance > minVariance) {
            const double correlation = covariance / std::sqrt(leftVariance * variance);
            if(correlation > bestCorrelation) {
                bestCorrelation = correlation;
                pixelDisparity = 0;
            }
        }

        // Store last column sums for sliding window
        double lastColSum = 0, lastColSqSum = 0;
        for(int y = 0; y < blockSize; y++) {
            const double* rightRow = rightData + y * rightStep;
            const double val = rightRow[blockSize - 1];
            lastColSum += val;
            lastColSqSum += val * val;
        }

        // Search disparities
        for(int disparity = 1; disparity <= maxDisparity; disparity++) {
            const int rightX = leftX - disparity;
            if (rightX < 0 || rightX + blockSize > combinedTSRight.cols) {
                break;
            }

            const double* newRightData = combinedTSRight.ptr<double>(topY) + rightX;

            // Compute new column sums
            double newColSum = 0, newColSqSum = 0;
            for(int y = 0; y < blockSize; y++) {
                const double* row = newRightData + y * rightStep;
                const double val = row[0];
                newColSum += val;
                newColSqSum += val * val;
            }

            // Update sliding statistics
            rightSum += newColSum - lastColSum;
            rightSumSq += newColSqSum - lastColSqSum;

            // Update last column for next iteration
            lastColSum = 0;
            lastColSqSum = 0;
            for(int y = 0; y < blockSize; y++) {
                const double* row = newRightData + y * rightStep;
                const double val = row[blockSize - 1];
                lastColSum += val;
                lastColSqSum += val * val;
            }

            // Compute covariance for current disparity
            sumXY = 0;
            for(int y = 0; y < blockSize; y++) {
                const double* leftRow = leftData + y * leftStep;
                const double* rightRow = newRightData + y * rightStep;
                for(int x = 0; x < blockSize; x++) {
                    sumXY += leftRow[x] * rightRow[x];
                }
            }

            variance = (rightSumSq - rightSum * rightSum / blockArea) / (blockArea - 1);
            covariance = (sumXY - leftSum * rightSum / blockArea) / (blockArea - 1);

            if(variance > minVariance) {
                const double correlation = covariance / std::sqrt(leftVariance * variance);
                if(correlation > bestCorrelation) {
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
        const int downsampling,
        const int searchBound,
        queue<dv::EventStore>& incomingLeftEvents,
        queue<cv::Mat>& incomingLeftImages,
        queue<cv::Mat>& incomingRightImages,
        queue<cv::Mat>& outgoingImages
    ) {
        while(true) {
            if(!incomingLeftImages.empty() && !incomingRightImages.empty() && !incomingLeftEvents.empty()) {
                dv::EventStore events = incomingLeftEvents.front();
                incomingLeftEvents.pop();
                cv::Mat leftImage = incomingLeftImages.front();
                incomingLeftImages.pop();
                cv::Mat rightImage = incomingRightImages.front();
                incomingRightImages.pop();

                vector<int> xCenters, yCenters;
                xCenters.reserve(downsampling);
                yCenters.reserve(downsampling);
                for(int i = 0; i < events.size(); i += downsampling) {
                    auto ev = events[i];
                    xCenters.push_back(ev.x());
                    yCenters.push_back(ev.y());
                }

                //cv::Mat left64, right64;
                //leftImage.convertTo(left64, CV_64F);
                //rightImage.convertTo(right64, CV_64F);

                vector<StereoBlockMatch> matchResult = stereoBlockMatchingParallel(
                    numThreads,
                    minVariance,
                    minCorrelation,
                    resolution,
                    halfBlockSize,
                    searchBound,
                    ref(leftImage),
                    ref(rightImage),
                    ref(xCenters),
                    ref(yCenters)
                );

                // render events based on horizontal or vertical orientation
                /*
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
                */

                cv::Mat vis = drawBlockMatchingResult(resolution, searchBound, matchResult);
                //cv::Mat eventVis = cv::Mat::zeros(resolution, CV_8UC3);
                /*
                for(int i = 0; i < xCenters.size(); i++) {
                    int x = xCenters[i];
                    int y = yCenters[i];
                    for(int i = max(0, x - 1); i < min(resolution.width, x + 2); i++) {
                        for(int j = max(0, y - 1); j < min(resolution.height, y + 2); j++) {
                            vis.at<cv::Vec3b>(j, i) = cv::Vec3b(255, 0, 0);
                        }
                    }
                }
                */
                //cv::Mat horizontalVis = drawBlockMatchingResult(resolution, searchBound, horizontalMatchResult);

                //vector<cv::Mat> outgoing = {eventVis, verticalVis, horizontalVis};

                outgoingImages.push(vis);
            }
            else {
                this_thread::sleep_for(2ms);
            }
        }
    }
}