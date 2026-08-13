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

#include <cameraCapture.h>
#include <imageRepresentation.h>
#include <tracking.h>
#include <mapping.h>

using namespace SlamDemo;

using namespace std::chrono_literals;
using namespace std;

int main(int argc, char* argv[])
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
    double minBlockVariance;
    double minBlockCorrelation;
    int sbmHalfBlockSize;
    int sbmNumThreads;
    int sbmSearchBound;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("calibration-json", po::value<std::string>(&calibJSONPath), "Camera calibration file written in JSON according to inivation standards")
        ("hot-pixels-dir", po::value<std::string>(&hotPixelsDir), "In which directory to find numpy files describing which camera pixels are hot")
        ("high-pass-us", po::value<int>(&highPassMicroseconds)->default_value(1000), "Background noise filter time constant in microseconds")
        ("low-pass-hz", po::value<double>(&lowPassHz)->default_value(500.0), "Low pass filter frequency (inverse of refractory period)")
        ("read-camera-ms", po::value<int>(&readCameraMilliseconds)->default_value(100), "Interval in milliseconds in which buffered data from the camera is sent to the rest of the system")
        ("time-surface-ms", po::value<int>(&timeSurfaceMilliseconds)->default_value(30), "Decay constant of the time surfaces in milliseconds")
        ("aa-patches-x", po::value<int>(&aaSurfacePatchesX)->default_value(4), "How many grid patches for adaptive accumulation, x axis")
        ("aa-patches-y", po::value<int>(&aaSurfacePatchesY)->default_value(3), "How many grid patches for adaptive accumulation, y axis")
        ("edge-feature-downsampling", po::value<int>(&edgeFeatureDownsampling)->default_value(10), "Minimum downsampling factor for converting (filtered) events into features for stereo matching")
        ("min-block-variance", po::value<double>(&minBlockVariance)->default_value(.05), "Minimum variance that a time surface patch must possess to be matched.")
        ("min-block-correlation", po::value<double>(&minBlockCorrelation)->default_value(.1), "Minimum correlation between two matched blocks in stereo block matching.")
        ("sbm-half-blocksize", po::value<int>(&sbmHalfBlockSize)->default_value(12), "Block side length used in SBM minus 1 divided by two.")
        ("sbm-num-threads", po::value<int>(&sbmNumThreads)->default_value(1), "Number of threads used for SBM.")
        ("sbm-search-bound", po::value<int>(&sbmSearchBound)->default_value(100), "Matching a block from the left camera will check at most this many blocks from the right camera.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    cout << "Camera calibration file: " << calibJSONPath << endl;
    cout << "Hot pixel directory: " << hotPixelsDir << endl;
    cout << "Background activity time constant: " << highPassMicroseconds << "us" << endl;
    cout << "Low pass frequency: " << lowPassHz << "Hz" << endl;
    cout << "Interval at which camera data is collected: " << readCameraMilliseconds << "ms" << endl;
    cout << "Time surface decay time constant: " << timeSurfaceMilliseconds << "ms" << endl;
    cout << "Adaptive accumulation patches X: " << aaSurfacePatchesX << endl;
    cout << "Adaptive accumulation patches Y: " << aaSurfacePatchesY << endl;
    cout << "Downsampling rate for edge feature detection: " << edgeFeatureDownsampling << endl;
    cout << "Minimum block variance for SBM: " << minBlockVariance << endl;
    cout << "Minimum block correlation for SBM: " << minBlockCorrelation << endl;
    cout << "SBM half block size: " << sbmHalfBlockSize << endl;
    cout << "SBM number of threads: " << sbmNumThreads << endl;
    cout << "SBM search bound: " << sbmSearchBound << endl;

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
    cv::namedWindow("Depth", cv::WINDOW_NORMAL);

    queue<dv::EventStore> leftEventQueue;
    queue<dv::EventStore> rightEventQueue;
    queue<vector<dv::IMU>> imuQueue;
    queue<cv::Mat> velocityVisQueue;
    //vector<cv::Affine3d> poseTrajectory;
    queue<tuple<vector<cv::Mat>, vector<dv::Event>, vector<dv::Event>>> leftImageToRender;
    queue<vector<cv::Mat>> rightImageToRender;
    queue<tuple<vector<cv::Mat>, vector<dv::Event>, vector<dv::Event>>> leftImageToMapping;
    queue<vector<cv::Mat>> rightImageToMapping;
    queue<vector<cv::Mat>> depthImageQueue;

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
    thread rightCameraCaptureThread(&rightCameraCapture,
                                    resolution,
                                    rightCameraCalib.name,
                                    hotPixelsDir + "/hot_pixels_right_x.npy",
                                    hotPixelsDir + "/hot_pixels_right_y.npy",
                                    highPassMicroseconds,
                                    lowPassHz,
                                    readCameraMilliseconds,
                                    ref(rightEventQueue));

    thread trackingThread(&imuPreintegration,
                          gyroBias,
                          accelBias,
                          ref(imuQueue),
                          ref(velocityVisQueue));

    thread leftImageRepresentationThread(&leftImageRepresentationLoop,
                                         resolution,
                                         aaSurfacePatchesX,
                                         aaSurfacePatchesY,
                                         edgeFeatureDownsampling,
                                         timeSurfaceMilliseconds,
                                         camMatL,
                                         distCoeffsL,
                                         ref(leftEventQueue),
                                         ref(leftImageToRender),
                                         ref(leftImageToMapping));
    thread rightImageRepresentationThread(&rightImageRepresentationLoop,
                                          resolution,
                                          timeSurfaceMilliseconds,
                                          camMatR,
                                          distCoeffsR,
                                          ref(rightEventQueue),
                                          ref(rightImageToRender),
                                          ref(rightImageToMapping));

    thread depthEstimationThread(
        &depthEstimationLoop,
        sbmNumThreads,
        minBlockVariance,
        minBlockCorrelation,
        resolution,
        sbmHalfBlockSize,
        sbmSearchBound,
        ref(leftImageToMapping),
        ref(rightImageToMapping),
        ref(depthImageQueue)
    );

    // Run the processing loop while both cameras are connected
    while (true) {
        if (!leftImageToRender.empty() && !rightImageToRender.empty()) {
            cv::Mat rightImage;
            cv::vconcat(rightImageToRender.front()[0], rightImageToRender.front()[1], rightImage);
            rightImageToRender.pop();
            auto leftqFront = leftImageToRender.front();
            leftImageToRender.pop();

            cv::Mat leftImage;
            cv::vconcat(get<0>(leftqFront)[0], get<0>(leftqFront)[1], leftImage);
            cv::vconcat(leftImage, get<0>(leftqFront)[2], leftImage);

            cv::imshow("Left", leftImage);
            cv::imshow("Right", rightImage);
        }
        if (!velocityVisQueue.empty()) {
            cv::Mat trajectoryVisualization = velocityVisQueue.front();
            velocityVisQueue.pop();
            cv::imshow("Trajectory", trajectoryVisualization);
        }
        if(!depthImageQueue.empty()) {
            auto depthImages = depthImageQueue.front();
            depthImageQueue.pop();
            cv::Mat depthImage;
            cv::vconcat(depthImages[0], depthImages[1], depthImage);
            cv::vconcat(depthImage, depthImages[2], depthImage);
            cv::imshow("Depth", depthImage);
        }
        // Wait for a small amount of time to avoid CPU overhaul
        cv::waitKey(2);
    }
    //*/
    return 0;
}