#include <opencv2/opencv.hpp>
#include <dv-processing/core/core.hpp>

#include <queue>
#include <iostream>

namespace SlamDemo {
    using namespace std;

    struct StereoBlockMatchingResult {
        vector<size_t> x;
        vector<size_t> y;
        vector<int> match;
        vector<double> correlation;
    };

    void singleBlockCrossCorrelation(
        const cv::Size resolution,
        const size_t halfBlockSize,
        const size_t searchBound,
        const size_t centerX,
        const size_t centerY,
        const cv::Mat& combinedTSLeft,
        const cv::Mat& combinedTSRight,
        cv::Scalar& leftVarianceAtBlock,
        vector<cv::Scalar>& rightVarianceAtDisparity,
        vector<cv::Scalar>& covarianceAtDisparity
    );

    void singleBlockSearch(
        const double minVariance,
        const double minCorrelation,
        cv::Scalar& leftVarianceAtBlock,
        vector<cv::Scalar>& rightVarianceAtDisparity,
        vector<cv::Scalar>& covarianceAtDisparity,
        int& match,
        double& bestCorrelation
    );

    void stereoBlockMatchingSequential(
        const double minVariance,
        const double minCorrelation,
        const cv::Size resolution,
        const size_t halfBlockSize,
        const size_t searchBound,
        const cv::Mat& combinedTSLeft,
        const cv::Mat& combinedTSRight,
        const vector<size_t>& xCenters,
        const vector<size_t>& yCenters,
        vector<cv::Scalar>& leftVariances,
        vector<vector<cv::Scalar>>& rightVariances,
        vector<vector<cv::Scalar>>& covariances,
        vector<int>& matches,
        vector<double>& correlations
    );

    StereoBlockMatchingResult returnStereoBlockMatching(
        const double minVariance,
        const double minCorrelation,
        const cv::Size resolution,
        const size_t halfBlockSize,
        const size_t searchBound,
        const cv::Mat& combinedTSLeft,
        const cv::Mat& combinedTSRight,
        const vector<dv::Event> eventsToMatch
    );

    cv::Mat drawBlockMatchingResult(
        const cv::Size resolution,
        const size_t searchBound,
        const StereoBlockMatchingResult sbmResult
    );

    void depthEstimationLoop(
        const double minVariance,
        const double minCorrelation,
        const cv::Size resolution,
        const size_t halfBlockSize,
        const size_t searchBound,
        queue<tuple<vector<cv::Mat>, vector<dv::Event>, vector<dv::Event>>>& incomingLeftData,
        queue<vector<cv::Mat>>& incomingRightData,
        queue<vector<cv::Mat>>& outgoingImages
    );
}