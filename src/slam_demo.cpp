#include <dv-processing/io/camera/discovery.hpp>
#include <dv-processing/visualization/event_visualizer.hpp>
#include <dv-processing/data/generate.hpp>
#include <dv-processing/noise/background_activity_noise_filter.hpp>
#include <dv-processing/noise/frequency_filters.hpp>
#include <dv-processing/core/core.hpp>
#include <dv-processing/camera/calibration_set.hpp>
#include <dv-processing/core/stereo_event_stream_slicer.hpp>
#include <dv-processing/depth/semi_dense_stereo_matcher.hpp>

#include <opencv2/highgui.hpp>

#include <iostream>
#include <condition_variable>
#include <future>

#include <cnpy.h>

#include <image_representation.cpp>

int main()
{
    using namespace std::chrono_literals;
    using namespace std;

    //*
    // Path to a stereo calibration file, replace with a file path on your local file system
    const string calibrationFilePath = "/home/kadile/Projects/fenrir/mimir_jr_calibration.json";

    // Load the calibration file
    auto calibration = dv::camera::CalibrationSet::LoadFromFile(calibrationFilePath);

    // It is expected that calibration file will have "C0" as the leftEventBuffer camera
    auto leftCameraCalib = calibration.getCameraCalibration("C0").value();

    // The second camera is assumed to be rightEventBuffer-side camera
    auto rightCameraCalib = calibration.getCameraCalibration("C1").value();

    // Open the stereo camera with camera names from calibration
    auto leftCamera  = dv::io::camera::open(leftCameraCalib.name);
    auto rightCamera = dv::io::camera::open(rightCameraCalib.name);

    //dv::io::camera::synchronizeAnyTwo(leftCamera, rightCamera);

    // Make sure both cameras support event stream output, throw an error otherwise
    if (!leftCamera->isEventStreamAvailable() || !rightCamera->isEventStreamAvailable()) {
        throw dv::exceptions::RuntimeError("Input camera does not provide an event stream.");
    }

    // Initialization of a stereo event sliver
    dv::EventStreamSlicer slicer_left;
    dv::EventStreamSlicer slicer_right;

    // Initialize a window to show previews of the output
    cv::namedWindow("Left", cv::WINDOW_NORMAL);
    cv::namedWindow("Right", cv::WINDOW_NORMAL);
    const cv::Size resolution = leftCameraCalib.resolution;
    dv::visualization::EventVisualizer visualizer_left(resolution);
    dv::visualization::EventVisualizer visualizer_right(resolution);

    // Apply color scheme configuration, these values can be modified to taste
    visualizer_left.setBackgroundColor(dv::visualization::colors::black);
    visualizer_left.setPositiveColor(dv::visualization::colors::blue);
    visualizer_left.setNegativeColor(dv::visualization::colors::green);
    visualizer_right.setBackgroundColor(dv::visualization::colors::black);
    visualizer_right.setPositiveColor(dv::visualization::colors::blue);
    visualizer_right.setNegativeColor(dv::visualization::colors::green);

    dv::EventPolarityFilter left_positive(true);
    dv::EventPolarityFilter left_negative(false);
    dv::EventPolarityFilter right_positive(true);
    dv::EventPolarityFilter right_negative(false);

    dv::Accumulator left_positive_ts(resolution);
    left_positive_ts.setMinPotential(0.f);
    left_positive_ts.setMaxPotential(1.f);
    left_positive_ts.setNeutralPotential(0.f);
    left_positive_ts.setEventContribution(1.f);
    left_positive_ts.setDecayFunction(dv::Accumulator::Decay::EXPONENTIAL);
    left_positive_ts.setDecayParam(1e+5);
    left_positive_ts.setIgnorePolarity(true);
    left_positive_ts.setSynchronousDecay(true);

    dv::Accumulator left_negative_ts(resolution);
    left_negative_ts.setMinPotential(0.f);
    left_negative_ts.setMaxPotential(1.f);
    left_negative_ts.setNeutralPotential(0.f);
    left_negative_ts.setEventContribution(1.f);
    left_negative_ts.setDecayFunction(dv::Accumulator::Decay::EXPONENTIAL);
    left_negative_ts.setDecayParam(1e+5);
    left_negative_ts.setIgnorePolarity(true);
    left_negative_ts.setSynchronousDecay(true);

    dv::Accumulator right_positive_ts(resolution);
    right_positive_ts.setMinPotential(0.f);
    right_positive_ts.setMaxPotential(1.f);
    right_positive_ts.setNeutralPotential(0.f);
    right_positive_ts.setEventContribution(1.f);
    right_positive_ts.setDecayFunction(dv::Accumulator::Decay::EXPONENTIAL);
    right_positive_ts.setDecayParam(3e+4);
    right_positive_ts.setIgnorePolarity(true);
    right_positive_ts.setSynchronousDecay(true);

    dv::Accumulator right_negative_ts(resolution);
    right_negative_ts.setMinPotential(0.f);
    right_negative_ts.setMaxPotential(1.f);
    right_negative_ts.setNeutralPotential(0.f);
    right_negative_ts.setEventContribution(1.f);
    right_negative_ts.setDecayFunction(dv::Accumulator::Decay::EXPONENTIAL);
    right_negative_ts.setDecayParam(3e+4);
    right_negative_ts.setIgnorePolarity(true);
    right_negative_ts.setSynchronousDecay(true);

    dv::SpeedInvariantTimeSurface speed_invariant_left(resolution);

    // Register a callback to be done at 20Hz
    slicer_left.doEveryTimeInterval(50ms, [&resolution, &left_positive_ts, &left_negative_ts, &left_positive, &left_negative, &speed_invariant_left](const auto &leftEvents) {

        auto speed_invariant_future = async(launch::async, [&]() {
            //speed_invariant_left.accept(leftEvents);
            //return speed_invariant_left.generateFrame();
            return adaptiveAccumulation(resolution, 8, 6, leftEvents.eigen());
        });

        // Start thread for negative time surface update
        auto negative_future = async(launch::async, [&]() {
            left_negative.accept(leftEvents);
            const auto negative = left_negative.generateEvents();
            left_negative_ts.accept(negative);
            return left_negative_ts.generateFrame();
        });

        // Update positive time surface on main thread
        left_positive.accept(leftEvents);
        const auto positive = left_positive.generateEvents();
        left_positive_ts.accept(positive);
        dv::Frame pos_frame = left_positive_ts.generateFrame();

        // Wait for negative thread to complete
        dv::Frame neg_frame = negative_future.get();
        cv::Mat speed_inv_frame = speed_invariant_future.get();

        // Combine and display
        vector<cv::Mat> images(3);
        images[0] = pos_frame.image;
        images[1] = neg_frame.image;
        images[2] = speed_inv_frame;
        cv::Mat left_image;
        cv::hconcat(images, left_image);
        cv::imshow("Left", left_image);
        cv::waitKey(2);
    });
    slicer_right.doEveryTimeInterval(50ms, [&right_positive_ts, &right_negative_ts, &right_positive, &right_negative](const auto &rightEvents) {

        // Start thread for negative time surface update
        auto negative_future = async(launch::async, [&]() {
            right_negative.accept(rightEvents);
            const auto negative = right_negative.generateEvents();
            right_negative_ts.accept(negative);
            return right_negative_ts.generateFrame();
        });

        // Update positive time surface on main thread
        right_positive.accept(rightEvents);
        const auto positive = right_positive.generateEvents();
        right_positive_ts.accept(positive);
        dv::Frame pos_frame = right_positive_ts.generateFrame();

        // Wait for negative thread to complete
        dv::Frame neg_frame = negative_future.get();

        // Combine and display
        vector<cv::Mat> images(2);
        images[0] = pos_frame.image;
        images[1] = neg_frame.image;
        cv::Mat right_image;
        cv::hconcat(images, right_image);
        cv::imshow("Right", right_image);
        cv::waitKey(2);
    });

    dv::noise::BackgroundActivityNoiseFilter high_pass_left(resolution, 10us);
    dv::noise::BackgroundActivityNoiseFilter high_pass_right(resolution, 10us);
    dv::noise::LowPassFilter low_pass_left(resolution, 500.0f);
    dv::noise::LowPassFilter low_pass_right(resolution, 500.0f);

    auto hot_pixels_left_x = cnpy::npy_load("/home/kadile/Projects/fenrir/hot_pixels_mimir_jr/hot_pixels_left_x.npy");
    auto hot_pixels_left_y = cnpy::npy_load("/home/kadile/Projects/fenrir/hot_pixels_mimir_jr/hot_pixels_left_y.npy");
    auto hot_pixels_right_x = cnpy::npy_load("/home/kadile/Projects/fenrir/hot_pixels_mimir_jr/hot_pixels_right_x.npy");
    auto hot_pixels_right_y = cnpy::npy_load("/home/kadile/Projects/fenrir/hot_pixels_mimir_jr/hot_pixels_right_y.npy");
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

    // Run the processing loop while both cameras are connected
    while (leftCamera->isRunning() && rightCamera->isRunning()) {
        // Read events from respective left / right cameras
        if (const auto raw = leftCamera->getNextEventBatch()) {
            high_pass_left.accept(*raw);
            const auto high = high_pass_left.generateEvents();
            low_pass_left.accept(high);
            const auto low = low_pass_left.generateEvents();
            mask_filter_left.accept(low);
            const auto masked = mask_filter_left.generateEvents();
            slicer_left.accept(low);
        }
        if (const auto raw = rightCamera->getNextEventBatch()) {
            high_pass_right.accept(*raw);
            const auto high = high_pass_right.generateEvents();
            low_pass_right.accept(high);
            const auto low = low_pass_right.generateEvents();
            mask_filter_right.accept(low);
            const auto masked = mask_filter_right.generateEvents();
            slicer_right.accept(masked);
        }

        // Wait for a small amount of time to avoid CPU overhaul
        //cv::waitKey(1);
    }
    //*/
    return 0;
}