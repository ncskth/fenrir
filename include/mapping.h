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
        const double minVariance,
        const double minCorrelation,
        const cv::Size resolution,
        const int halfBlockWidth,
        const int halfBlockHeight,
        const int searchBound,
        const int centerX,
        const int centerY,
        const cv::Mat& combinedTSLeft,
        const cv::Mat& combinedTSRight
    );

    vector<StereoBlockMatch> stereoBlockMatchingSequential(
        const double minVariance,
        const double minCorrelation,
        const cv::Size resolution,
        const int halfBlockWidth,
        const int halfBlockHeight,
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
        const int halfBlockWidth,
        const int halfBlockHeight,
        const int searchBound,
        const cv::Mat& combinedTSLeft,
        const cv::Mat& combinedTSRight,
        const vector<int>& xCenters,
        const vector<int>& yCenters
    );

    cv::Mat drawBlockMatchingResult(
        const cv::Size resolution,
        const int searchBound,
        const vector<StereoBlockMatch>& sbmResult
    );

    void depthEstimationLoop(
        const int numThreads,
        const double minVariance,
        const double minCorrelation,
        const cv::Size resolution,
        const int halfBlockWidth,
        const int halfBlockHeight,
        const int downsampling,
        const int searchBound,
        queue<dv::EventStore>& incomingLeftEvents,
        queue<cv::Mat>& incomingLeftImages,
        queue<cv::Mat>& incomingRightImages,
        queue<cv::Mat>& outgoingImages
    );
}