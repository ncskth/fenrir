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

#include <boost/program_options.hpp>

#include <cnpy.h>

#include <iostream>
#include <condition_variable>
#include <future>
#include <string>

#include <cameraCapture.cpp>
#include <tracking.cpp>
#include <imageRepresentation.cpp>

using namespace SlamDemo;

using namespace std::chrono_literals;
using namespace std;

int main()
{

    namespace po = boost::program_options;

    string calibJSONPath;
    string hotPixelsDir;
    int highPassMicroseconds;
    double lowPassHz;
    int readCameraMilliseconds;
    int timeSurfaceMilliseconds;
    int aaSurfacePatchesX;
    int aaSurfacePatchesY;
    int edgeFeatureDownsampling;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("calibration-json", po::value<std::string>(&calibJSONPath), "Camera calibration file written in JSON according to inivation standards")
        ("hot-pixels-dir", po::value<std::string>(&hotPixelsDir), "In which directory to find numpy files describing which camera pixels are hot")
        ("high-pass-us", po::value<int>(&highPassMicroseconds)->default_value(1000), "Background noise filter time constant in microseconds")
        ("low-pass-hz", po::value<double>(&lowPassHz)->default_value(500.0), "Low pass filter frequency (inverse of refractory period)")
        ("read-camera-ms", po::value<int>(&readCameraMilliseconds)->default_value(20), "Interval in milliseconds in which buffered data from the camera is sent to the rest of the system")
        ("time-surface-ms", po::value<int>(&timeSurfaceMilliseconds)->default_value(30), "Decay constant of the time surfaces in milliseconds")
        ("aa-patches-x", po::value<int>(&aaSurfacePatchesX)->default_value(8), "How many grid patches for adaptive accumulation, x axis")
        ("aa-patches-y", po::value<int>(&aaSurfacePatchesY)->default_value(6), "How many grid patches for adaptive accumulation, y axis")
        ("edge-feature-downsampling", po::value<int>(&edgeFeatureDownsampling)->default_value(100), "Minimum downsampling factor for converting (filtered) events into features for stereo matching");

    // Load the calibration file
    auto calibration = dv::camera::CalibrationSet::LoadFromFile(calibJSONPath);

    // It is expected that calibration file will have "C0" as the leftEventBuffer camera
    auto leftCameraCalib = calibration.getCameraCalibration("C0").value();
    const cv::Size resolution = leftCameraCalib.resolution;
    cv::Matx33f camMatL = leftCameraCalib.getCameraMatrix();
    vector<float> distCoeffsL = leftCameraCalib.distortion;

    // The second camera is assumed to be rightEventBuffer-side camera
    auto rightCameraCalib = calibration.getCameraCalibration("C1").value();
    cv::Matx33f camMatR = rightCameraCalib.getCameraMatrix();
    vector<float> distCoeffsR = rightCameraCalib.distortion;

    auto imuCalib = calibration.getImuCalibration("S0").value();
    cv::Point3f gyroBias = imuCalib.omegaOffsetAvg;
    cv::Point3f accelBias = imuCalib.accOffsetAvg;

    // Initialize a window to show previews of the output
    cv::namedWindow("Left", cv::WINDOW_NORMAL);
    cv::namedWindow("Right", cv::WINDOW_NORMAL);
    cv::namedWindow("Trajectory", cv::WINDOW_NORMAL);

    queue<dv::EventStore> leftEventQueue;
    queue<dv::EventStore> rightEventQueue;
    queue<vector<dv::IMU>> imuQueue;
    queue<cv::Mat> velocityVisQueue;
    //vector<cv::Affine3d> poseTrajectory;
    queue<tuple<vector<cv::Mat>, vector<dv::Event>, vector<dv::Event>>> leftImageQueue;
    queue<vector<cv::Mat>> rightImageQueue;

    cv::Mat trajectoryVisualization = cv::Mat::zeros(400, 400, CV_8UC1);

    thread leftCameraCaptureThread(&leftCameraCapture,
                                   resolution,
                                   leftCameraCalib.name,
                                   hotPixelsDir + "/hot_pixels_left_x.npy",
                                   hotPixelsDir + "/hot_pixels_left_y.npy",
                                   highPassMicroseconds,
                                   lowPassHz,
                                   readCameraMilliseconds,
                                   ref(leftEventQueue),
                                   ref(imuQueue));
    thread rightCameraCaptureThread(&leftCameraCapture,
                                    resolution,
                                    leftCameraCalib.name,
                                    hotPixelsDir + "/hot_pixels_right_x.npy",
                                    hotPixelsDir + "/hot_pixels_right_y.npy",
                                    highPassMicroseconds,
                                    lowPassHz,
                                    readCameraMilliseconds,
                                    ref(leftEventQueue),
                                    ref(imuQueue));

    thread trackingThread(&imuPreintegration,
                          gyroBias,
                          accelBias,
                          ref(imuQueue),
                          ref(velocityVisQueue));

    thread leftImageRepresentationThread(&leftImageRepresentationLoop,
                                         resolution,
                                         aaSurfacePatchesX,
                                         aaSurfacePatchesY,
                                         timeSurfaceMilliseconds,
                                         camMatL,
                                         distCoeffsL,
                                         ref(leftEventQueue),
                                         ref(leftImageQueue));
    thread rightImageRepresentationThread(&rightImageRepresentationLoop,
                                          resolution,
                                          timeSurfaceMilliseconds,
                                          camMatR,
                                          distCoeffsR,
                                          ref(rightEventQueue),
                                          ref(rightImageQueue));

    // Run the processing loop while both cameras are connected
    while (true) {
        if (!leftImageQueue.empty()) {
            auto leftqFront = leftImageQueue.front();
            leftImageQueue.pop();

            cv::Mat leftImage;
            cv::hconcat(get<0>(leftqFront)[0], get<0>(leftqFront)[1], leftImage);
            cv::hconcat(leftImage, get<0>(leftqFront)[2], leftImage);
            cv::Mat leftImageBGR;
            cv::cvtColor(leftImage, leftImageBGR, cv::COLOR_GRAY2BGR);

            for(auto ev : get<1>(leftqFront)) {
                leftImageBGR.at<cv::Vec3b>(ev.y(), ev.x()) = cv::Vec3b(255, 0, 0);
            }
            for(auto ev : get<2>(leftqFront)) {
                leftImageBGR.at<cv::Vec3b>(ev.y(), ev.x()) = cv::Vec3b(0, 0, 255);
            }

            cv::imshow("Left", leftImageBGR);
        }
        if (!rightImageQueue.empty()) {
            cv::Mat rightImage;
            cv::hconcat(rightImageQueue.front()[0], rightImageQueue.front()[1], rightImage);
            rightImageQueue.pop();
            cv::imshow("Right", rightImage);
        }
        if (!velocityVisQueue.empty()) {
            cv::Mat trajectoryVisualization = velocityVisQueue.front();
            velocityVisQueue.pop();
            cv::imshow("Trajectory", trajectoryVisualization);
        }
        // Wait for a small amount of time to avoid CPU overhaul
        cv::waitKey(2);
    }
    //*/
    return 0;
}