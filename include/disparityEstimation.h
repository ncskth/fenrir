#pragma once

#include <opencv2/opencv.hpp>
#include <dv-processing/core/core.hpp>

#include <queue>
#include <iostream>
#include <future>

namespace Fenrir {
    using namespace std;

    struct StereoBlockMatch {
        int32_t x;
        int32_t y;
        int32_t pixelDisparity;
    };

    tuple<cv::Mat, cv::Mat> edgeKernels(int width);

    bool isHorizontalEdge(const cv::Mat &img,
                              const cv::Mat &kernelx,
                              const cv::Mat &kernely,
                              int y, int x);

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
        const cv::Mat &combinedTSRight
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
        const vector<tuple<int, int>> &centers,
        const int start,
        const int end
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
        const vector<tuple<int, int>> &centers
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
        queue<vector<tuple<int, int>>> &incomingLeftEvents,
        cv::Mat &leftImage,
        cv::Mat &rightImage,
        queue<vector<StereoBlockMatch>> &outgoingEstimates
    );
}