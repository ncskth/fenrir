#include <dv-processing/core/core.hpp>

#include <opencv2/opencv.hpp>

#include <iostream>
#include <queue>
#include <future>

namespace SlamDemo {
    using namespace std;

    cv::Mat adaptiveAccumulation(
        cv::Size resolution,
        int x_patches,
        int y_patches,
        const dv::EventStore events
    );

    cv::Mat leftImageRepresentation(
        const cv::Size resolution,
        const int aaPatchesX,
        const int aaPatchesY,
        const int downsampleFactor,
        const cv::Matx33f cameraMatrix,
        const vector<float> distortionCoefficients,
        const int tsDecayMs,
        vector<int64_t>& lastPosts,
        vector<int64_t>& lastNegts,
        dv::EventStore &events
    );

    cv::Mat imageRepresentation(
        const cv::Size resolution,
        const cv::Matx33f cameraMatrix,
        const vector<float> distortionCoefficients,
        const int tsDecayMs,
        vector<int64_t>& lastPosts,
        vector<int64_t>& lastNegts,
        dv::EventStore &events
    );

    void leftImageRepresentationLoop(
        const cv::Size resolution,
        const int aaPatchesX,
        const int aaPatchesY,
        const int downsampleFactor,
        const int timeSurfaceMilliseconds,
        const cv::Matx33f &cameraMatrix,
        const vector<float> &distortionCoefficients,
        queue<dv::EventStore> &incomingEvents,
        queue<cv::Mat> &outgoingImages1,
        queue<cv::Mat> &outgoingImages2
    );

    void imageRepresentationLoop(
        const cv::Size resolution,
        const int timeSurfaceMilliseconds,
        const cv::Matx33f &cameraMatrix,
        const vector<float> &distortionCoefficients,
        queue<dv::EventStore> &incomingEvents,
        queue<cv::Mat> &outgoingImages1,
        queue<cv::Mat> &outgoingImages2
    );
}