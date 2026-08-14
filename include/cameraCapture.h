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

    void rightCameraCapture(
        const cv::Size resolution,
        const string serial,
        const string hotPixelXFile,
        const string hotPixelYFile,
        const int accumulatorTimeConstant,
        const double accumulatorGain,
        const int sendIntervalMilliseconds,
        queue<cv::Mat>& outgoingImages1,
        queue<cv::Mat>& outgoingImages2
    );

    void leftCameraCapture(
        const cv::Size resolution,
        const string serial,
        const string hotPixelXFile,
        const string hotPixelYFile,
        const int accumulatorTimeConstant,
        const double accumulatorGain,
        const int sendIntervalMilliseconds,
        queue<dv::EventStore>& outgoingEvents,
        queue<cv::Mat>& outgoingImages1,
        queue<cv::Mat>& outgoingImages2,
        queue<vector<dv::IMU>>& outgoingIMU
    );
}