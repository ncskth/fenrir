#include <mapping.h>


namespace SlamDemo {
    using namespace std::chrono_literals;
    using namespace std;

    void singleBlockCrossCorrelation(
        const cv::Size resolution,
        const int halfBlockSize,
        const int searchBound,
        const int centerX,
        const int centerY,
        const cv::Mat& combinedTSLeft,
        const cv::Mat& combinedTSRight,
        cv::Scalar& leftVarianceAtBlock,
        vector<cv::Scalar>& rightVarianceAtDisparity,
        vector<cv::Scalar>& covarianceAtDisparity
    ) {
        int blockSize = 2*halfBlockSize + 1;
        int blockArea = blockSize*blockSize;
        int leftX = centerX - halfBlockSize;
        int topY = centerY - halfBlockSize;
        cv::Rect leftRect(leftX, topY, blockSize, blockSize);
        cout << "Left rectangle: " << leftRect << endl;

        cv::Mat leftPatch = combinedTSLeft(leftRect);
        cv::Scalar leftMean;
        cv::mean(leftPatch, leftMean);
        cv::Mat znPatch = leftPatch - leftMean;
        leftVarianceAtBlock = cv::sum(znPatch.mul(znPatch))/((double)blockArea);

        int numSearches = min(searchBound, 1 + leftX);

        cv::Rect rightRect = leftRect;
        cv::Mat rightPatch = combinedTSRight(rightRect);
        cv::Scalar sumIntensity = cv::sum(rightPatch);
        cv::Scalar sumSquaredIntensity = cv::sum(rightPatch.mul(rightPatch));
        cv::Scalar variance = ((double)blockArea*sumSquaredIntensity - sumIntensity*sumIntensity)/((double)blockArea*blockArea*blockArea);
        rightVarianceAtDisparity.push_back(variance);
        cv::Scalar covariance = (cv::sum(leftPatch.mul(rightPatch)) - leftMean*sumIntensity)/((double)blockArea);
        covarianceAtDisparity.push_back(covariance);
        cv::Mat lastPatch = rightPatch;

        cout << "Covariance vector length before search: " << covarianceAtDisparity.size() << endl;
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

            variance = ((double)blockArea*sumSquaredIntensity - sumIntensity*sumIntensity)/((double)blockArea*blockArea*blockArea);
            rightVarianceAtDisparity.push_back(variance);
            covariance = (cv::sum(leftPatch.mul(rightPatch)) - leftMean*sumIntensity)/((double)blockArea);
            covarianceAtDisparity.push_back(covariance);
        }
        cout << "Covariance vector length after search: " << covarianceAtDisparity.size() << endl;
    }

    void singleBlockSearch(
        const double minVariance,
        const double minCorrelation,
        cv::Scalar& leftVarianceAtBlock,
        vector<cv::Scalar>& rightVarianceAtDisparity,
        vector<cv::Scalar>& covarianceAtDisparity,
        int& match,
        double& bestCorrelation
    ) {
        // -1 for no match
        match = -1;
        bestCorrelation = 0.;
        if(leftVarianceAtBlock[0] < minVariance) {
            cout << "Left patch did not pass disparity check!" << endl;
            return;
        }

        for(int disparity = 0; disparity < covarianceAtDisparity.size(); disparity++) {
            if(rightVarianceAtDisparity[disparity][0] > minVariance) {
                double corr = covarianceAtDisparity[disparity][0]/(sqrt(leftVarianceAtBlock[0]*rightVarianceAtDisparity[disparity][0]));
                cout << "Correlation: " << corr << endl;
                if(corr > minCorrelation && corr > bestCorrelation) {
                    match = (int)disparity;
                    corr = bestCorrelation;
                }
            } else {
                cout << "Right patch did not pass disparity check!" << endl;
            }
        }
    }

    void stereoBlockMatchingSequential(
        const double minVariance,
        const double minCorrelation,
        const cv::Size resolution,
        const int halfBlockSize,
        const int searchBound,
        const cv::Mat& combinedTSLeft,
        const cv::Mat& combinedTSRight,
        const vector<int>& xCenters,
        const vector<int>& yCenters,
        vector<cv::Scalar>& leftVariances,
        vector<vector<cv::Scalar>>& rightVariances,
        vector<vector<cv::Scalar>>& covariances,
        vector<int>& matches,
        vector<double>& correlations
    ) {
        int numBlocks = xCenters.size();

        for(int i = 0; i < numBlocks; i++) {
            singleBlockCrossCorrelation(
                resolution,
                halfBlockSize,
                searchBound,
                xCenters[i],
                yCenters[i],
                combinedTSLeft,
                combinedTSRight,
                ref(leftVariances[i]),
                ref(rightVariances[i]),
                ref(covariances[i])
            );
            singleBlockSearch(
                minVariance,
                minCorrelation,
                ref(leftVariances[i]),
                ref(rightVariances[i]),
                ref(covariances[i]),
                ref(matches[i]),
                ref(correlations[i])
            );
        }
    }

    StereoBlockMatchingResult returnStereoBlockMatching(
        const double minVariance,
        const double minCorrelation,
        const cv::Size resolution,
        const int halfBlockSize,
        const int searchBound,
        const cv::Mat& combinedTSLeft,
        const cv::Mat& combinedTSRight,
        const vector<dv::Event> eventsToMatch
    ) {
        // avoid matching events which are at the edge of the image
        vector<int> xCenters, yCenters;
        for(auto ev: eventsToMatch) {
            if(ev.x() >= halfBlockSize && ev.y() >= halfBlockSize && ev.x() < resolution.width - halfBlockSize && ev.y() < resolution.height - halfBlockSize) {
                xCenters.push_back((int)ev.x());
                yCenters.push_back((int)ev.y());
            }
        }
        int numEventsToMatch = xCenters.size();

        // initialize all the data for matching
        vector<cv::Scalar> leftVariances(numEventsToMatch);
        vector<double> correlation(numEventsToMatch);
        vector<int> matches(numEventsToMatch);
        vector<vector<cv::Scalar>> rightVariances(numEventsToMatch), covariances(numEventsToMatch);
        for(int i = 0; i < numEventsToMatch; i++) {
            rightVariances[i].reserve(searchBound);
            covariances[i].reserve(searchBound);
        }

        stereoBlockMatchingSequential(
            minVariance,
            minCorrelation,
            resolution,
            halfBlockSize,
            searchBound,
            ref(combinedTSLeft),
            ref(combinedTSRight),
            ref(xCenters),
            ref(yCenters),
            ref(leftVariances),
            ref(rightVariances),
            ref(covariances),
            ref(matches),
            ref(correlation)
        );

        return {xCenters, yCenters, matches, correlation};
    }

    cv::Mat drawBlockMatchingResult(
        const cv::Size resolution,
        const int searchBound,
        const StereoBlockMatchingResult sbmResult
    ){

        // render horizontal and vertical depth estimations separately
        // use hue to indicate disparity, value to indicate confidence
        cv::Mat visHSV = cv::Mat::zeros(resolution, CV_8UC3);
        cv::Mat visBGR;
        for(int i = 0; i < sbmResult.x.size(); i++) {
            if(sbmResult.match[i] > -1) {
                int x = sbmResult.x[i];
                int y = sbmResult.y[i];
                uint8_t hue = (uint8_t)(179.*sbmResult.match[i]/searchBound);
                uint8_t val = (uint8_t)(255.*sbmResult.correlation[i]*sbmResult.correlation[i]);
                for(int i = x - 1; i < x + 2; i++) {
                    for(int j = y - 1; j < y + 2; j++) {
                        visHSV.at<cv::Vec3b>(j, i) = cv::Vec3b(hue, 255, val);
                    }
                }
            }
        }
        cv::cvtColor(visHSV, visBGR, cv::COLOR_HSV2BGR);

        return visBGR;
    }

    void depthEstimationLoop(
        const double minVariance,
        const double minCorrelation,
        const cv::Size resolution,
        const int halfBlockSize,
        const int searchBound,
        queue<tuple<vector<cv::Mat>, vector<dv::Event>, vector<dv::Event>>>& incomingLeftData,
        queue<vector<cv::Mat>>& incomingRightData,
        queue<vector<cv::Mat>>& outgoingImages
    ) {
        while(true) {
            if(!incomingLeftData.empty() && !incomingRightData.empty()) {
                auto leftImageDatum = incomingLeftData.front();
                incomingLeftData.pop();
                auto rightImageDatum = incomingRightData.front();
                incomingRightData.pop();

                cv::Mat negTSLeft = get<0>(leftImageDatum)[0];
                cv::Mat posTSLeft = get<0>(leftImageDatum)[1];
                cv::Mat negTSRight = rightImageDatum[0];
                cv::Mat posTSRight = rightImageDatum[1];
                vector<dv::Event> verticalEvents = get<1>(leftImageDatum);
                vector<dv::Event> horizontalEvents = get<2>(leftImageDatum);

                // combined positive and negative time surfaces into [-1, 1] double-valued array
                cv::Mat negTSLeftf, posTSLeftf, negTSRightf, posTSRightf, combinedTSLeft, combinedTSRight;
                negTSLeft.convertTo(negTSLeftf, CV_64FC1);
                posTSLeft.convertTo(posTSLeftf, CV_64FC1);
                combinedTSLeft = negTSLeftf + posTSLeftf - 255.;
                combinedTSLeft /= 255.;
                negTSRight.convertTo(negTSRightf, CV_64FC1);
                negTSRight.convertTo(negTSRightf, CV_64FC1);
                combinedTSRight = negTSRightf + posTSRightf - 255.;
                combinedTSRight /= 255.;

                StereoBlockMatchingResult verticalMatchResult = returnStereoBlockMatching(
                    minVariance,
                    minCorrelation,
                    resolution,
                    halfBlockSize,
                    searchBound,
                    ref(combinedTSLeft),
                    ref(combinedTSRight),
                    verticalEvents
                );
                StereoBlockMatchingResult horizontalMatchResult = returnStereoBlockMatching(
                    minVariance,
                    minCorrelation,
                    resolution,
                    halfBlockSize,
                    searchBound,
                    ref(combinedTSLeft),
                    ref(combinedTSRight),
                    verticalEvents
                );

                // render events based on horizontal or vertical orientation
                cv::Mat eventVis = cv::Mat::zeros(resolution, CV_8UC3);
                for(auto ev : verticalEvents) {
                    for(int i = max(0, (int)ev.x() - 1); i < min(resolution.width, (int)ev.x() + 2); i++) {
                        for(int j = max(0, (int)ev.y() - 1); j < min(resolution.height, (int)ev.y() + 2); j++) {
                            eventVis.at<cv::Vec3b>(j, i) = cv::Vec3b(255, 0, 0);
                        }
                    }
                }
                for(auto ev : horizontalEvents) {
                    for(int i = max(0, (int)ev.x() - 1); i < min(resolution.width, (int)ev.x() + 2); i++) {
                        for(int j = max(0, (int)ev.y() - 1); j < min(resolution.height, (int)ev.y() + 2); j++) {
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