#include <dv-processing/io/camera/discovery.hpp>
#include <dv-processing/visualization/event_visualizer.hpp>
#include <dv-processing/data/generate.hpp>
#include <dv-processing/noise/background_activity_noise_filter.hpp>
#include <dv-processing/noise/frequency_filters.hpp>
#include <dv-processing/core/core.hpp>
#include <dv-processing/camera/calibration_set.hpp>
#include <dv-processing/core/stereo_event_stream_slicer.hpp>
#include <dv-processing/depth/semi_dense_stereo_matcher.hpp>

#include <opencv2/viz.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <condition_variable>
#include <future>

#include <cnpy.h>

#include <cameraCapture.cpp>
#include <imageRepresentation.cpp>

using namespace std::chrono_literals;
using namespace std;

int main()
{
    //*
    // Path to a stereo calibration file, replace with a file path on your local file system
    const string calibrationFilePath = "/home/kadile/Projects/fenrir/mimir_jr_calibration.json";

    // Load the calibration file
    auto calibration = dv::camera::CalibrationSet::LoadFromFile(calibrationFilePath);

    // It is expected that calibration file will have "C0" as the leftEventBuffer camera
    auto leftCameraCalib = calibration.getCameraCalibration("C0").value();
    const cv::Size resolution = leftCameraCalib.resolution;
    cv::Matx33f camMatL = leftCameraCalib.getCameraMatrix();
    vector<float> distCoeffsL = leftCameraCalib.distortion;

    // The second camera is assumed to be rightEventBuffer-side camera
    auto rightCameraCalib = calibration.getCameraCalibration("C1").value();
    cv::Matx33f camMatR = rightCameraCalib.getCameraMatrix();
    vector<float> distCoeffsR = rightCameraCalib.distortion;

    // Initialize a window to show previews of the output
    cv::namedWindow("Left", cv::WINDOW_NORMAL);
    cv::namedWindow("Right", cv::WINDOW_NORMAL);
    cv::viz::Viz3d window("Trajectory");

    queue<dv::EventStore> leftEventQueue;
    queue<dv::EventStore> rightEventQueue;
    queue<cv::Affine3d> poseQueue;
    vector<cv::Affine3d> poseTrajectory;
    queue<vector<cv::Mat>> leftImageQueue;
    queue<vector<cv::Mat>> rightImageQueue;

    thread CameraCaptureThread(&cameraCaptureCallback,
                               resolution,
                               leftCameraCalib.name,
                               rightCameraCalib.name,
                               "/home/kadile/Projects/fenrir/hot_pixels_mimir_jr",
                               ref(leftEventQueue),
                               ref(rightEventQueue),
                               ref(poseQueue));

    thread leftImageRepresentationThread(&imageRepresentationCallback,
                                         true,
                                         resolution,
                                         camMatL,
                                         distCoeffsL,
                                         ref(leftEventQueue),
                                         ref(leftImageQueue));
    thread rightImageRepresentationThread(&imageRepresentationCallback,
                                          false,
                                          resolution,
                                          camMatR,
                                          distCoeffsR,
                                          ref(rightEventQueue),
                                          ref(rightImageQueue));

    // Run the processing loop while both cameras are connected
    while (true) {
        if (!leftImageQueue.empty()) {
            cv::Mat leftImage;
            cv::hconcat(leftImageQueue.front()[0], leftImageQueue.front()[1], leftImage);
            cv::hconcat(leftImage, leftImageQueue.front()[2], leftImage);
            leftImageQueue.pop();
            cv::imshow("Left", leftImage);
        }
        if (!rightImageQueue.empty()) {
            cv::Mat rightImage;
            cv::hconcat(rightImageQueue.front()[0], rightImageQueue.front()[1], rightImage);
            rightImageQueue.pop();
            cv::imshow("Right", rightImage);
        }
        if (!poseQueue.empty()) {
            cv::Affine3d pose = poseQueue.front();
            poseQueue.pop();
            poseTrajectory.push_back(pose);
            cv::viz::WTrajectory trajectory(poseTrajectory, cv::viz::WTrajectory::PATH, 1.0, cv::viz::Color::green());
            window.showWidget("Tajectory", trajectory);
            window.spinOnce(1, true);
            if(poseTrajectory.size() > 2000) {
                poseTrajectory.erase(poseTrajectory.begin(), poseTrajectory.begin() + 1000);
            }
        }
        // Wait for a small amount of time to avoid CPU overhaul
        cv::waitKey(1);
    }
    //*/
    return 0;
}