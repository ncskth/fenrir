#include <dv-processing/io/camera/discovery.hpp>
#include <dv-processing/visualization/event_visualizer.hpp>
#include <dv-processing/data/generate.hpp>
#include <dv-processing/noise/background_activity_noise_filter.hpp>
#include <dv-processing/noise/frequency_filters.hpp>
#include <dv-processing/core/core.hpp>
#include <dv-processing/camera/calibration_set.hpp>

#include <opencv2/highgui.hpp>

#include <queue>
#include <iostream>
#include <condition_variable>
#include <future>

#include <cnpy.h>

namespace SlamDemo {

using namespace std::chrono_literals;
using namespace std;

mutex leftCameraMutex;
condition_variable leftCameraCondition;
bool leftCameraReady = false;
mutex rightCameraMutex;
condition_variable rightCameraCondition;
bool rightCameraReady = false;

void rightCameraCapture(const cv::Size resolution,
                        const string serial,
                        const string hotPixelXFile,
                        const string hotPixelYFile,
                        const int highPassMicroseconds,
                        const double lowPassHz,
                        const int sendIntervalMilliseconds,
                        queue<dv::EventStore>& outgoingEvents
                        ) {

    // Open the stereo camera with camera names from calibration
    auto camera  = dv::io::camera::open(serial);

    // Make sure both cameras support event stream output, throw an error otherwise
    if (!camera->isEventStreamAvailable()) {
        throw dv::exceptions::RuntimeError("Input camera does not provide an event stream.");
    }

    dv::noise::BackgroundActivityNoiseFilter highPass(resolution, highPassMicroseconds*1us);
    dv::noise::LowPassFilter lowPass(resolution, lowPassHz);

    auto hotPixelsX = cnpy::npy_load(hotPixelXFile);
    auto hotPixelsY = cnpy::npy_load(hotPixelYFile);
    cv::Mat mask(resolution, CV_8UC1, cv::Scalar(255));
    for(size_t i = 0; i < hotPixelsX.num_vals; i++) {
        int x = hotPixelsX.data<int>()[i];
        int y = hotPixelsY.data<int>()[i];
        mask.at<uchar>(x, y) = 0;
    }
    dv::EventMaskFilter maskFilter(mask.t());

    dv::EventStore eventBuffer;

    rightCameraReady = true;
    unique_lock lk(rightCameraMutex);
    rightCameraCondition.wait(lk, []{return leftCameraReady;});
    rightCameraReady = false;

    auto lastQPush = chrono::high_resolution_clock::now();

    while (camera->isRunning()) {
        if (const auto raw = camera->getNextEventBatch()) {
            highPass.accept(*raw);
            const auto high = highPass.generateEvents();
            lowPass.accept(high);
            const auto low = lowPass.generateEvents();
            maskFilter.accept(low);
            const auto masked = maskFilter.generateEvents();
            eventBuffer.add(masked);
        }

        auto now = chrono::high_resolution_clock::now();
        if (now - lastQPush > sendIntervalMilliseconds * 1ms) {
            outgoingEvents.push(eventBuffer);
            eventBuffer = dv::EventStore();
            lastQPush = now;

            rightCameraReady = true;
            unique_lock lk(rightCameraMutex);
            rightCameraCondition.wait(lk, []{return leftCameraReady;});
            rightCameraReady = false;
        }
    }
}

void leftCameraCapture(const cv::Size resolution,
                       const string serial,
                       const string hotPixelXFile,
                       const string hotPixelYFile,
                       const int highPassMicroseconds,
                       const double lowPassHz,
                       const int sendIntervalMilliseconds,
                       queue<dv::EventStore>& outgoingEvents,
                       queue<vector<dv::IMU>>& outgoingIMU
                       ) {

    // Open the stereo camera with camera names from calibration
    auto camera  = dv::io::camera::open(serial);

    // Make sure both cameras support event stream output, throw an error otherwise
    if (!camera->isEventStreamAvailable()) {
        throw dv::exceptions::RuntimeError("Input camera does not provide an event stream.");
    }
    if (!camera->isImuStreamAvailable()) {
        throw dv::exceptions::RuntimeError("Input camera does not provide an IMU stream.");
    }

    dv::noise::BackgroundActivityNoiseFilter highPass(resolution, highPassMicroseconds*1us);
    dv::noise::LowPassFilter lowPass(resolution, lowPassHz);

    auto hotPixelsX = cnpy::npy_load(hotPixelXFile);
    auto hotPixelsY = cnpy::npy_load(hotPixelYFile);
    cv::Mat mask(resolution, CV_8UC1, cv::Scalar(255));
    for(size_t i = 0; i < hotPixelsX.num_vals; i++) {
        int x = hotPixelsX.data<int>()[i];
        int y = hotPixelsY.data<int>()[i];
        mask.at<uchar>(x, y) = 0;
    }
    dv::EventMaskFilter maskFilter(mask.t());

    dv::EventStore eventBuffer;
    vector<dv::IMU> imuBuffer;

    leftCameraReady = true;
    unique_lock lk(leftCameraMutex);
    leftCameraCondition.wait(lk, []{return rightCameraReady;});
    leftCameraReady = false;

    auto lastQPush = chrono::high_resolution_clock::now();

    while (camera->isRunning()) {
        if (const auto raw = camera->getNextEventBatch()) {
            highPass.accept(*raw);
            const auto high = highPass.generateEvents();
            lowPass.accept(high);
            const auto low = lowPass.generateEvents();
            maskFilter.accept(low);
            const auto masked = maskFilter.generateEvents();
            eventBuffer.add(masked);
        }

        if (const auto imuBatch = camera->getNextImuBatch()) {
            imuBuffer.insert(imuBuffer.end(), imuBatch->begin(), imuBatch->end());
        }

        auto now = chrono::high_resolution_clock::now();
        if (now - lastQPush > sendIntervalMilliseconds * 1ms) {
            outgoingEvents.push(eventBuffer);
            eventBuffer = dv::EventStore();
            outgoingIMU.push(imuBuffer);
            imuBuffer.clear();
            lastQPush = now;

            leftCameraReady = true;
            unique_lock lk(leftCameraMutex);
            leftCameraCondition.wait(lk, []{return rightCameraReady;});
            leftCameraReady = false;
        }
    }
}
}