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

using namespace std::chrono_literals;
using namespace std;

void cameraCaptureCallback(const cv::Size resolution,
                           const string leftSerial,
                           const string rightSerial,
                           const string hotPixelDir,
                           queue<dv::EventStore>& outgoingLeftEvents,
                           queue<dv::EventStore>& outgoingRightEvents) {

    // Open the stereo camera with camera names from calibration
    auto leftCamera  = dv::io::camera::open(leftSerial);
    auto rightCamera = dv::io::camera::open(rightSerial);

    // Make sure both cameras support event stream output, throw an error otherwise
    if (!leftCamera->isEventStreamAvailable() || !rightCamera->isEventStreamAvailable()) {
        throw dv::exceptions::RuntimeError("Input camera does not provide an event stream.");
    }

    dv::noise::BackgroundActivityNoiseFilter high_pass_left(resolution, 10us);
    dv::noise::BackgroundActivityNoiseFilter high_pass_right(resolution, 10us);
    dv::noise::LowPassFilter low_pass_left(resolution, 500.0f);
    dv::noise::LowPassFilter low_pass_right(resolution, 500.0f);

    auto hot_pixels_left_x = cnpy::npy_load(hotPixelDir + "/hot_pixels_left_x.npy");
    auto hot_pixels_left_y = cnpy::npy_load(hotPixelDir + "/hot_pixels_left_y.npy");
    auto hot_pixels_right_x = cnpy::npy_load(hotPixelDir + "/hot_pixels_right_x.npy");
    auto hot_pixels_right_y = cnpy::npy_load(hotPixelDir + "/hot_pixels_right_y.npy");
    cv::Mat mask_left(resolution, CV_8UC1, cv::Scalar(255));
    cv::Mat mask_right(resolution, CV_8UC1, cv::Scalar(255));
    for(size_t i = 0; i < hot_pixels_left_x.num_vals; i++) {
        int x = hot_pixels_left_x.data<int>()[i];  // Assuming int coordinates
        int y = hot_pixels_left_y.data<int>()[i];
        mask_left.at<uchar>(x, y) = 0;  // (row, col) = (y, x)
    }

    for(size_t i = 0; i < hot_pixels_right_x.num_vals; i++) {
        int x = hot_pixels_right_x.data<int>()[i];
        int y = hot_pixels_right_y.data<int>()[i];
        mask_right.at<uchar>(x, y) = 0;
    }
    dv::EventMaskFilter mask_filter_left(mask_left.t());
    dv::EventMaskFilter mask_filter_right(mask_right.t());

    dv::EventStore leftEventBuffer;
    dv::EventStore rightEventBuffer;

    auto lastLeft = chrono::high_resolution_clock::now();
    auto lastRight = chrono::high_resolution_clock::now();

    while (leftCamera->isRunning() && rightCamera->isRunning()) {
        // Read events from respective left / right cameras
        if (const auto raw = leftCamera->getNextEventBatch()) {
            high_pass_left.accept(*raw);
            const auto high = high_pass_left.generateEvents();
            low_pass_left.accept(high);
            const auto low = low_pass_left.generateEvents();
            mask_filter_left.accept(low);
            const auto masked = mask_filter_left.generateEvents();
            leftEventBuffer.add(masked);
        }
        if (const auto raw = rightCamera->getNextEventBatch()) {
            high_pass_right.accept(*raw);
            const auto high = high_pass_right.generateEvents();
            low_pass_right.accept(high);
            const auto low = low_pass_right.generateEvents();
            mask_filter_right.accept(low);
            const auto masked = mask_filter_right.generateEvents();
            rightEventBuffer.add(masked);
        }
        auto now = chrono::high_resolution_clock::now();
        if (now - lastLeft > 33ms) {
            outgoingLeftEvents.push(leftEventBuffer);
            leftEventBuffer = dv::EventStore();
            lastLeft = now;
        }
        //if (leftEventBuffer.size() > 200000) {
        //    outgoingLeftEvents.push(leftEventBuffer);
        //    leftEventBuffer = dv::EventStore();
        //}
        if (now - lastRight > 33ms) {
            outgoingRightEvents.push(rightEventBuffer);
            rightEventBuffer = dv::EventStore();
            lastRight = now;
        }
        //if (rightEventBuffer.size() > 200000) {
        //    outgoingRightEvents.push(rightEventBuffer);
        //    rightEventBuffer = dv::EventStore();
        //}
    }
}