#include <dv-processing/io/camera/discovery.hpp>
#include <dv-processing/core/core.hpp>
#include <dv-processing/camera/calibration_set.hpp>

#include <opencv2/viz.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <boost/program_options.hpp>

#include <cnpy.h>

#include <iostream>
#include <condition_variable>
#include <future>
#include <string>

#include <cameraCapture.h>
#include <imageRepresentation.h>
#include <tracking.h>
#include <disparityEstimation.h>
#include <sender.hpp>
#include <visualization.h>
#include <calibData.h>

using namespace Fenrir;

using namespace std::chrono_literals;
using namespace std;

int main(int argc, char* argv[])
{

    namespace po = boost::program_options;

    string calibJSONPath;
    string hotPixelsDir;
    string clientAddr;
    int highPassMicroseconds;
    double lowPassHz;
    int readCameraMilliseconds;
    int eventAccumulatorMilliseconds;
    double eventAccumulatorGain;
    int aaSurfacePatchesX;
    int aaSurfacePatchesY;
    double minBlockVariance;
    double minBlockCorrelation;
    int maxEventsInBuffer;
    int sbmEventDownsampling;
    int sbmHalfBlockWidth;
    int sbmHalfBlockHeight;
    int sbmNumThreads;
    int sbmSearchBound;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("calibration-json", po::value<std::string>(&calibJSONPath), "Camera calibration file written in JSON according to inivation standards")
        ("hot-pixels-dir", po::value<std::string>(&hotPixelsDir), "In which directory to find numpy files describing which camera pixels are hot")
        ("client-addr", po::value<std::string>(&clientAddr)->default_value(""), "If empty, visualize on the local machine, otherwise, send visualization frames to this IP address")
        ("high-pass-us", po::value<int>(&highPassMicroseconds)->default_value(1000), "Background noise filter time constant in microseconds")
        ("low-pass-hz", po::value<double>(&lowPassHz)->default_value(500.0), "Low pass filter frequency (inverse of refractory period)")
        ("read-camera-ms", po::value<int>(&readCameraMilliseconds)->default_value(100), "Interval in milliseconds in which buffered data from the camera is sent to the rest of the system")
        ("event-accumulator-ms", po::value<int>(&eventAccumulatorMilliseconds)->default_value(1000), "Decay constant of the event accumulator in milliseconds")
        ("event-accumulator-gain", po::value<double>(&eventAccumulatorGain)->default_value(0.15), "Contribution of individual events to the accumulator")
        ("aa-patches-x", po::value<int>(&aaSurfacePatchesX)->default_value(4), "How many grid patches for adaptive accumulation, x axis")
        ("aa-patches-y", po::value<int>(&aaSurfacePatchesY)->default_value(3), "How many grid patches for adaptive accumulation, y axis")
        ("sbm-min-variance", po::value<double>(&minBlockVariance)->default_value(0.05), "Minimum variance that a time surface patch must possess to be matched.")
        ("sbm-event-downsampling", po::value<int>(&sbmEventDownsampling)->default_value(50), "Rate at which we downsample events to get blocks for SBM")
        ("sbm-min-correlation", po::value<double>(&minBlockCorrelation)->default_value(.7), "Minimum correlation between two matched blocks in stereo block matching.")
        ("max-events-in-buffer", po::value<int>(&maxEventsInBuffer)->default_value(12000), "Maximum number of events which the left camera sends into the system at a time")
        ("sbm-half-block-width", po::value<int>(&sbmHalfBlockWidth)->default_value(24), "Block width used in SBM minus 1 divided by two.")
        ("sbm-half-block-height", po::value<int>(&sbmHalfBlockHeight)->default_value(6), "Block height used in SBM minus 1 divided by two.")
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
    cout << "Event accumulator decay time constant: " << eventAccumulatorMilliseconds << "ms" << endl;
    cout << "Event accumulator gain per event: " << eventAccumulatorGain << endl;
    cout << "Adaptive accumulation patches X: " << aaSurfacePatchesX << endl;
    cout << "Adaptive accumulation patches Y: " << aaSurfacePatchesY << endl;
    cout << "SBM minimum block variance: " << minBlockVariance << endl;
    cout << "SBM minimum block correlation: " << minBlockCorrelation << endl;
    cout << "Maximum number of events in buffer: " << maxEventsInBuffer << endl;
    cout << "SBM downsampling rate of events to obtain block centers: " << sbmEventDownsampling << endl;
    cout << "SBM half block width: " << sbmHalfBlockWidth << endl;
    cout << "SBM half block height: " << sbmHalfBlockHeight << endl;
    cout << "SBM number of threads: " << sbmNumThreads << endl;
    cout << "SBM search bound: " << sbmSearchBound << endl;

    CalibData calibData = loadCalibData(calibJSONPath);

    /*
    auto imuCalib = calibration.getImuCalibration("S0").value();
    cv::Point3f gyroBias = imuCalib.omegaOffsetAvg;
    cv::Point3f accelBias = imuCalib.accOffsetAvg;
    */

    queue<vector<tuple<int, int>>> leftEventsToMap;
    queue<vector<dv::IMU>> imuQueue;
    queue<cv::Mat> velocityVisQueue;
    queue<vector<StereoBlockMatch>> depthEstimateQueue;

    cv::Mat leftImage = 0.5*cv::Mat::ones(calibData.height, calibData.width, CV_64FC1);
    cv::Mat rightImage = 0.5*cv::Mat::ones(calibData.height, calibData.width, CV_64FC1);
    vector<int64_t> leftTimestamps(calibData.resolution.area());
    vector<int64_t> rightTimestamps(calibData.resolution.area());
    for(int i = 0; i < calibData.resolution.area(); i++) {
        leftTimestamps[i] = 0;
        rightTimestamps[i] = 0;
    }

    cv::Mat trajectoryVisualization = cv::Mat::zeros(400, 400, CV_8UC1);

    thread leftCameraCaptureThread(
        &leftCameraCapture,
        calibData.resolution,
        calibData.leftCameraName,
        hotPixelsDir + "/hot_pixels_left_x.npy",
        hotPixelsDir + "/hot_pixels_left_y.npy",
        highPassMicroseconds,
        eventAccumulatorMilliseconds,
        eventAccumulatorGain,
        readCameraMilliseconds,
        maxEventsInBuffer,
        calibData.leftUndistortRectifyMap1,
        calibData.leftUndistortRectifyMap2,
        ref(leftEventsToMap),
        ref(leftImage),
        ref(leftTimestamps),
        ref(imuQueue)
    );
    thread rightCameraCaptureThread(
        &rightCameraCapture,
        calibData.resolution,
        calibData.rightCameraName,
        hotPixelsDir + "/hot_pixels_right_x.npy",
        hotPixelsDir + "/hot_pixels_right_y.npy",
        highPassMicroseconds,
        eventAccumulatorMilliseconds,
        eventAccumulatorGain,
        readCameraMilliseconds,
        calibData.rightUndistortRectifyMap1,
        calibData.rightUndistortRectifyMap2,
        ref(rightImage),
        ref(rightTimestamps)
    );

    //thread trackingThread(
    //    &imuPreintegration,
    //    gyroBias,
    //    accelBias,
    //    ref(imuQueue),
    //    ref(velocityVisQueue)
    //);

    thread depthEstimationThread(
        &depthEstimationLoop,
        sbmNumThreads,
        minBlockVariance,
        minBlockCorrelation,
        calibData.resolution,
        sbmHalfBlockWidth,
        sbmHalfBlockHeight,
        sbmEventDownsampling,
        sbmSearchBound,
        ref(leftEventsToMap),
        ref(leftImage),
        ref(rightImage),
        ref(depthEstimateQueue)
    );

    if(clientAddr.empty()) {

        cv::namedWindow("Depth Demo", cv::WINDOW_NORMAL);

        cv::Mat depthColorKey = drawDepthColorKey(
            calibData.focalLength,
            calibData.baseline,
            sbmSearchBound
        );
        cv::Mat image = cv::Mat::zeros(960, 1280, CV_8UC3);
        // Place the color key at [0:480, 640:840].
        depthColorKey.copyTo(image(cv::Rect(640, 0, depthColorKey.cols, depthColorKey.rows)));

        while (true) {

            if (!depthEstimateQueue.empty()) {
                auto depthResults = depthEstimateQueue.front();
                depthEstimateQueue.pop();

                cv::Mat leftImageToRender = 255*leftImage;
                cv::Mat rightImageToRender = 255*rightImage;
                cv::Mat leftImageU8, rightImageU8;
                leftImageToRender.convertTo(leftImageU8, CV_8UC1);
                rightImageToRender.convertTo(rightImageU8, CV_8UC1);

                updateDepthDemoFrame(
                    sbmSearchBound,
                    depthResults,
                    leftImageU8,
                    rightImageU8,
                    ref(image)
                );

                // write to display
                cv::imshow("Depth Demo", image);
                cv::waitKey(2);
            }

            if (!imuQueue.empty()) {
                imuQueue.pop();
            }

            // Wait for a small amount of time to avoid CPU overhaul
            this_thread::sleep_for(chrono::milliseconds(1));
        }
    }
    else {
        auto sender = Sender("tcp://" + clientAddr + ":5555");

        while (true) {
            if (!depthEstimateQueue.empty()) {
                auto depthResults = depthEstimateQueue.front();
                depthEstimateQueue.pop();

                cv::Mat leftImageToRender = 255*leftImage;
                cv::Mat rightImageToRender = 255*rightImage;
                cv::Mat leftImageU8, rightImageU8;
                leftImageToRender.convertTo(leftImageU8, CV_8UC1);
                rightImageToRender.convertTo(rightImageU8, CV_8UC1);

                sender.sendFullDepthFrame(leftImageU8, rightImageU8, depthResults);
            }
            if (!imuQueue.empty()) {
                imuQueue.pop();
            }
            // Wait for a small amount of time to avoid CPU overhaul
            this_thread::sleep_for(chrono::milliseconds(2));
        }
    }
    return 0;
}