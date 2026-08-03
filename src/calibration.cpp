#include <dv-processing/io/camera/discovery.hpp>
#include <dv-processing/visualization/event_visualizer.hpp>
#include <dv-processing/data/generate.hpp>
#include <dv-processing/noise/background_activity_noise_filter.hpp>
#include <dv-processing/noise/frequency_filters.hpp>
#include <dv-processing/core/core.hpp>
#include <dv-processing/camera/calibration_set.hpp>

#include <opencv2/highgui.hpp>
#include <cnpy.h>

#include <string>

using namespace std::chrono_literals;
using namespace std;

int main()
{

    // Open any camera
    auto leftCamera = dv::io::camera::open("DXM00089");
    auto rightCamera = dv::io::camera::open("DXM00090");
    cv::Size resolution(640, 480);
    string hotPixelDir = "/home/kadile/Projects/fenrir/hot_pixels_mimir_jr";

    // Make sure both cameras support event stream output, throw an error otherwise
    if (!leftCamera->isEventStreamAvailable() || !rightCamera->isEventStreamAvailable())
    {
        throw dv::exceptions::RuntimeError("Input camera does not provide an event stream.");
    }

    dv::noise::BackgroundActivityNoiseFilter high_pass_left(resolution, 10ms);
    dv::noise::BackgroundActivityNoiseFilter high_pass_right(resolution, 10ms);
    dv::noise::BandPassFilter low_pass_left(resolution, 49.f, 51.f);
    dv::noise::BandPassFilter low_pass_right(resolution, 49.f, 51.f);

    auto hot_pixels_left_x = cnpy::npy_load(hotPixelDir + "/hot_pixels_left_x.npy");
    auto hot_pixels_left_y = cnpy::npy_load(hotPixelDir + "/hot_pixels_left_y.npy");
    auto hot_pixels_right_x = cnpy::npy_load(hotPixelDir + "/hot_pixels_right_x.npy");
    auto hot_pixels_right_y = cnpy::npy_load(hotPixelDir + "/hot_pixels_right_y.npy");
    cv::Mat mask_left(resolution, CV_8UC1, cv::Scalar(255));
    cv::Mat mask_right(resolution, CV_8UC1, cv::Scalar(255));
    for (size_t i = 0; i < hot_pixels_left_x.num_vals; i++)
    {
        int x = hot_pixels_left_x.data<int>()[i]; // Assuming int coordinates
        int y = hot_pixels_left_y.data<int>()[i];
        mask_left.at<uchar>(x, y) = 0; // (row, col) = (y, x)
    }

    for (size_t i = 0; i < hot_pixels_right_x.num_vals; i++)
    {
        int x = hot_pixels_right_x.data<int>()[i];
        int y = hot_pixels_right_y.data<int>()[i];
        mask_right.at<uchar>(x, y) = 0;
    }
    dv::EventMaskFilter mask_filter_left(mask_left.t());
    dv::EventMaskFilter mask_filter_right(mask_right.t());

    // Initialize an accumulator with some resolution
    dv::visualization::EventVisualizer visualizer_left(*leftCamera->getEventResolution());
    dv::visualization::EventVisualizer visualizer_right(*rightCamera->getEventResolution());
    dv::Accumulator accumulator_left(*leftCamera->getEventResolution());
    dv::Accumulator accumulator_right(*rightCamera->getEventResolution());

    // Apply configuration, these values can be modified to taste
    accumulator_left.setMinPotential(0.f);
    accumulator_left.setMaxPotential(1.f);
    accumulator_left.setNeutralPotential(0.5f);
    accumulator_left.setEventContribution(0.15f);
    accumulator_left.setDecayFunction(dv::Accumulator::Decay::EXPONENTIAL);
    accumulator_left.setDecayParam(1e+6);
    accumulator_left.setIgnorePolarity(false);
    accumulator_left.setSynchronousDecay(false);

    accumulator_right.setMinPotential(0.f);
    accumulator_right.setMaxPotential(1.f);
    accumulator_right.setNeutralPotential(0.5f);
    accumulator_right.setEventContribution(0.15f);
    accumulator_right.setDecayFunction(dv::Accumulator::Decay::EXPONENTIAL);
    accumulator_right.setDecayParam(1e+6);
    accumulator_right.setIgnorePolarity(false);
    accumulator_right.setSynchronousDecay(false);

    // Initialize a preview window
    cv::namedWindow("Left", cv::WINDOW_NORMAL);
    cv::namedWindow("Right", cv::WINDOW_NORMAL);

    // Initialize a slicer
    dv::EventStreamSlicer slicer_left;
    dv::EventStreamSlicer slicer_right;

    vector<cv::Mat> leftImages;
    vector<cv::Mat> rightImages;

    // Register a callback every 33 milliseconds
    slicer_left.doEveryTimeInterval(100ms, [&accumulator_left, &visualizer_left, &leftImages](const dv::EventStore &events)
                                    {
        // Pass events into the accumulator and generate a preview frame
        //visualizer_left.accept(events);
        cv::Mat frame = visualizer_left.generateImage(events);

        // Show the accumulated image
        cv::imshow("Left", frame);
        leftImages.push_back(frame);
        cv::waitKey(2); });
    slicer_right.doEveryTimeInterval(100ms, [&accumulator_right, &visualizer_right, &rightImages](const dv::EventStore &events)
                                     {
        // Pass events into the accumulator and generate a preview frame
        //visualizer_right.accept(events);
        cv::Mat frame = visualizer_right.generateImage(events);

        // Show the accumulated image
        cv::imshow("Right", frame);
        rightImages.push_back(frame);
        cv::waitKey(2); });

    auto last = chrono::high_resolution_clock::now();

    // Run the event processing while the camera is connected
    while (leftCamera->isRunning() && leftCamera->isRunning() && chrono::high_resolution_clock::now() - last < 30s)
    {
        // Receive events, check if anything was received
        if (const auto raw = leftCamera->getNextEventBatch())
        {
            high_pass_left.accept(*raw);
            const auto high = high_pass_left.generateEvents();
            low_pass_left.accept(high);
            const auto low = low_pass_left.generateEvents();
            mask_filter_left.accept(low);
            const auto masked = mask_filter_left.generateEvents();
            slicer_left.accept(masked);
        }
        if (const auto raw = rightCamera->getNextEventBatch())
        {
            high_pass_right.accept(*raw);
            const auto high = high_pass_right.generateEvents();
            low_pass_right.accept(high);
            const auto low = low_pass_right.generateEvents();
            mask_filter_right.accept(low);
            const auto masked = mask_filter_right.generateEvents();
            slicer_right.accept(masked);
        }
    }

    cv::destroyAllWindows();

    // Termination criteria
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);

    // Prepare object points
    cv::Mat objp(6 * 9, 3, CV_32FC1);
    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 9; j++)
        {
            objp.at<float>(i * 7 + j, 0) = j;
            objp.at<float>(i * 7 + j, 1) = i;
            objp.at<float>(i * 7 + j, 2) = 0;
        }
    }

    // Arrays to store object points and image points
    std::vector<std::vector<cv::Point3f>> objpoints;
    std::vector<std::vector<cv::Point2f>> imgpoints;

    // Get all jpg images
    std::vector<cv::String> images;
    cv::glob("*.jpg", images);

    for (const cv::Mat gray : leftImages)
    {
        //cv::Mat img = cv::imread(fname);
        //cv::Mat gray;
        //cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        cout << "Checking frame." << endl;

        // Find chessboard corners
        std::vector<cv::Point2f> corners;
        bool ret = cv::findChessboardCorners(gray, cv::Size(9, 6), corners);

        // If found, add object points, image points (after refining)
        if (ret)
        {
            std::vector<cv::Point3f> objp_vec;
            for (int i = 0; i < 6 * 9; i++)
            {
                objp_vec.push_back(cv::Point3f(objp.at<float>(i, 0),
                                               objp.at<float>(i, 1),
                                               objp.at<float>(i, 2)));
            }
            objpoints.push_back(objp_vec);

            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);
            imgpoints.push_back(corners);

            // Draw and display corners
            cv::Mat display = gray.clone();
            cv::drawChessboardCorners(display, cv::Size(9, 6), corners, ret);
            cv::imshow("img", display);
            cv::waitKey(500);
        } else {
            cout << "No Chessboard!" << endl;
            cv::imshow("img", gray);
            cv::waitKey(500);
        }
    }

    return 0;
}