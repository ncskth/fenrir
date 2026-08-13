#include <opencv2/opencv.hpp>
#include <dv-processing/core/core.hpp>

#include <queue>
#include <iostream>

namespace SlamDemo {
    using namespace std;

    struct StereoBlockMatchingResult {
        vector<int> x;
        vector<int> y;
        vector<int> match;
        vector<double> correlation;
    };

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
        vector<double>& correlations,
        int start, int end
    );

    void stereoBlockMatchingParallel(
        const int numThreads,
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
    );

    StereoBlockMatchingResult returnStereoBlockMatching(
        const int numThreads,
        const double minVariance,
        const double minCorrelation,
        const cv::Size resolution,
        const int halfBlockSize,
        const int searchBound,
        const cv::Mat& combinedTSLeft,
        const cv::Mat& combinedTSRight,
        const vector<dv::Event> eventsToMatch
    );

    cv::Mat drawBlockMatchingResult(
        const cv::Size resolution,
        const int searchBound,
        const StereoBlockMatchingResult sbmResult
    );

    void depthEstimationLoop(
        int numThreads,
        const double minVariance,
        const double minCorrelation,
        const cv::Size resolution,
        const int halfBlockSize,
        const int searchBound,
        queue<tuple<vector<cv::Mat>, vector<dv::Event>, vector<dv::Event>>>& incomingLeftData,
        queue<vector<cv::Mat>>& incomingRightData,
        queue<vector<cv::Mat>>& outgoingImages
    );
}