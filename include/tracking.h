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

namespace SlamDemo {
    using namespace std;

    tuple<cv::Mat, double> initalRotationFromGravity(vector<dv::IMU> imuReadings, cv::Point3f bias);

    void imuPreintegration(
        const cv::Point3f gyroBias,
        const cv::Point3f accelBias,
        queue<vector<dv::IMU>>& incomingIMU,
        queue<cv::Mat>& outgoingVelocityVis
    );
}