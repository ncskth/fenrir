#include <dv-processing/io/camera/discovery.hpp>
#include <dv-processing/visualization/event_visualizer.hpp>
#include <dv-processing/data/generate.hpp>
#include <dv-processing/noise/background_activity_noise_filter.hpp>
#include <dv-processing/noise/frequency_filters.hpp>
#include <dv-processing/core/core.hpp>
#include <dv-processing/camera/calibration_set.hpp>

#include <opencv2/opencv.hpp>

#include <queue>
#include <iostream>
#include <string>
#include <barrier>

#include <cnpy.h>

namespace SlamDemo {
    using namespace std;

    void updateImageAndTimestamps(
        const double decay,
        const double gain,
        const int width,
        const cv::Mat& undistortRectifyMap1,
        const cv::Mat& undistortRectifyMap2,
        const dv::EventStore& events,
        cv::Mat& image,
        vector<int64_t>& timestamps
    );

    void rightCameraCapture(
        const cv::Size resolution,
        const string serial,
        const string hotPixelXFile,
        const string hotPixelYFile,
        const int highPassMicroseconds,
        const int accumulatorTimeConstant,
        const double accumulatorGain,
        const int sendIntervalMilliseconds,
        const cv::Mat undistortRectifyMat1,
        const cv::Mat undistortRectifyMat2,
        cv::Mat& image,
        vector<int64_t>& timestamps
    );

    void leftCameraCapture(
        const cv::Size resolution,
        const string serial,
        const string hotPixelXFile,
        const string hotPixelYFile,
        const int highPassMicroseconds,
        const int accumulatorTimeConstant,
        const double accumulatorGain,
        const int sendIntervalMilliseconds,
        const int maxEventsInBuffer,
        const cv::Mat undistortRectifyMat1,
        const cv::Mat undistortRectifyMat2,
        queue<vector<tuple<int, int>>>& outgoingEvents,
        cv::Mat& image,
        vector<int64_t>& timestamps,
        queue<vector<dv::IMU>>& outgoingIMU
    );
}