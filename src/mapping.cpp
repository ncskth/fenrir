#include <opencv2/highgui.hpp>

#include <queue>
#include <iostream>
#include <condition_variable>
#include <future>

#include <cnpy.h>


namespace SlamDemo {
    using namespace std;

    struct blockCovarianceResult {
        vector<vector<cv::Scalar>> covariancesAtDisparity;
        vector<cv::Scalar> leftVarianceAtBlock;
        vector<vector<cv::Scalar>> rightVarianceAtBlock;
    };

    blockCovarianceResult fastBlockCrossCovariance(
        const cv::Size resolution,
        const size_t halfBlockSize,
        const size_t searchBound,
        const cv::Mat leftNegTS,
        const cv::Mat leftPosTS,
        const cv::Mat rightNegTS,
        const cv::Mat rightPosTS) {
        cv::Mat leftTS, leftTSf, rightTS, rightTSf;
        vector<cv::Mat> leftChannels = {leftNegTS, leftPosTS};
        vector<cv::Mat> rightChannels = {leftNegTS, leftPosTS};
        cv::merge(leftChannels, leftTS);
        leftTS.convertTo(leftTSf, CV_32FC2);
        leftTSf /= 255.;
        cv::merge(rightChannels, rightTS);
        rightTS.convertTo(rightTSf, CV_32FC2);
        rightTSf /= 255.;

        size_t blockSize = 2*halfBlockSize + 1;
        size_t blockArea = blockSize*blockSize;
        size_t blocksPerRow = resolution.width - blockSize;
        size_t blocksPerCol = resolution.height - blockSize;
        size_t totalBlocks = blocksPerRow*blocksPerCol;

        vector<vector<cv::Scalar>> covariancesAtBlock(totalBlocks), rightVariancesAtBlock(totalBlocks);
        vector<cv::Scalar> leftMeanAtBlock(totalBlocks), leftVarianceAtBlock(totalBlocks);

        for(size_t x = 0; x < blocksPerRow; x += 1) {
            for(size_t y = 0; y < blocksPerCol; y += 1) {
                size_t ix = x + blocksPerRow*y;
                cv::Rect rect(x, y, blockSize, blockSize);

                cv::Mat patch = leftTSf(rect);

                cv::Scalar mean;
                cv::mean(patch, mean);
                cv::Mat znPatch = patch - mean;
                cv::Scalar leftVariance = cv::sum(znPatch.mul(znPatch));
                leftVariance /= (double)blockArea - 1;
                leftVarianceAtBlock[ix] = leftVariance;

                size_t numSearches = min(searchBound, 1 + x);
                covariancesAtBlock[ix].reserve(numSearches);
                rightVariancesAtBlock[ix].reserve(numSearches);

                cv::Mat rightPatch = rightTSf(rect);
                cv::Scalar sumIntensity = cv::sum(rightPatch);
                cv::Scalar sumSquaredIntensity = cv::sum(rightPatch.mul(rightPatch));
                cv::Scalar variance = ((double)blockArea*sumSquaredIntensity - sumIntensity*sumIntensity)/((double)blockArea*blockArea*blockArea);
                rightVariancesAtBlock[ix].push_back(variance);
                cv::Scalar covariance = (cv::sum(patch.mul(rightPatch)) - mean*sumIntensity)/((double)blockArea);
                covariancesAtBlock[ix].push_back(covariance);
                cv::Mat lastPatch = rightPatch;

                for(size_t disparity = 1; disparity < numSearches; disparity++) {
                    rect = cv::Rect(x - disparity, y, blockSize, blockSize);
                    rightPatch = rightTSf(rect);
                    cv::Mat lastCol = lastPatch.col(blockSize - 1);
                    sumIntensity -= cv::sum(lastCol);
                    cv::Mat nextCol = rightPatch.col(0);
                    sumIntensity += cv::sum(nextCol);
                    sumSquaredIntensity -= cv::sum(lastCol.mul(lastCol));
                    sumSquaredIntensity += cv::sum(nextCol.mul(nextCol));

                    lastPatch = rightPatch;

                    variance = ((double)blockArea*sumSquaredIntensity - sumIntensity*sumIntensity)/((double)blockArea*blockArea*blockArea);
                    rightVariancesAtBlock[ix].push_back(variance);
                    covariance = (cv::sum(patch.mul(rightPatch)) - mean*sumIntensity)/((double)blockArea);
                    covariancesAtBlock[ix].push_back(covariance);
                }
            }
        }

        return {covariancesAtBlock, leftVarianceAtBlock, rightVariancesAtBlock};
    }
}