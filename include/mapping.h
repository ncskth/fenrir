#include <opencv2/opencv.hpp>
#include <dv-processing/core/core.hpp>

#include <queue>
#include <iostream>
#include <future>

namespace SlamDemo {
    using namespace std;

    struct StereoBlockMatch {
        int x;
        int y;
        int pixelDisparity;
        double correlation;
    };

    StereoBlockMatch matchSingleBlock(
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
    );

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
        const vector<int>& yCenters,
        vector<cv::Scalar>& leftVariances,
        vector<vector<cv::Scalar>>& rightVariances,
        vector<vector<cv::Scalar>>& covariances,
        vector<int>& matches,
        vector<double>& correlations
    );

    cv::Mat drawBlockMatchingResult(
        const cv::Size resolution,
        const int searchBound,
        const vector<StereoBlockMatch>& sbmResult
    );

    tuple<double, double> sobelAtPoint(cv::Mat img, int y, int x);

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
    );
}