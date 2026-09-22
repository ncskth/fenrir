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

    tuple<cv::Mat, cv::Mat> edgeKernels(int width);

    bool isHorizontalEdge(const cv::Mat &img,
                              const cv::Mat &kernelx,
                              const cv::Mat &kernely,
                              int y, int x);
    void drawSingleMatch(
        const int searchBound,
        const StereoBlockMatch& match,
        cv::Mat& imageHSV);

    StereoBlockMatch matchSingleBlock(
        const double minVariance,
        const double minCorrelation,
        const cv::Size resolution,
        const int halfBlockWidth,
        const int halfBlockHeight,
        const int searchBound,
        const int centerX,
        const int centerY,
        const cv::Mat &kernelx,
        const cv::Mat &kernely,
        const cv::Mat &combinedTSLeft,
        const cv::Mat &combinedTSRight,
        cv::Mat& imgHSV
    );

    vector<StereoBlockMatch> stereoBlockMatchingSequential(
        const double minVariance,
        const double minCorrelation,
        const cv::Size resolution,
        const int halfBlockWidth,
        const int halfBlockHeight,
        const int searchBound,
        const cv::Mat &kernelx,
        const cv::Mat &kernely,
        const cv::Mat &combinedTSLeft,
        const cv::Mat &combinedTSRight,
        const vector<int> &xCenters,
        const vector<int> &yCenters,
        const int start,
        const int end,
        cv::Mat& imgHSV
    );

    vector<vector<StereoBlockMatch>> stereoBlockMatchingParallel(
        const int numThreads,
        const double minVariance,
        const double minCorrelation,
        const cv::Size resolution,
        const int halfBlockWidth,
        const int halfBlockHeight,
        const int searchBound,
        const cv::Mat &kernelx,
        const cv::Mat &kernely,
        const cv::Mat &combinedTSLeft,
        const cv::Mat &combinedTSRight,
        const vector<int> &xCenters,
        const vector<int> &yCenters,
        cv::Mat& imgHSV
    );

    cv::Mat drawBlockMatchingResult(
        const cv::Size resolution,
        const int searchBound,
        const vector<vector<StereoBlockMatch>> &sbmResult
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
        queue<vector<dv::Event>> &incomingLeftEvents,
        cv::Mat &leftImage,
        cv::Mat &rightImage,
        queue<cv::Mat> &outgoingImages
    );
}