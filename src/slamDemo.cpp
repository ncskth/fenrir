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
#include <mapping.h>

using namespace SlamDemo;

using namespace std::chrono_literals;
using namespace std;

cv::Mat drawDepthColorKey(const cv::Mat& Q, int sbmSearchBound) {
    // Get focal length and baseline from Q
    double f = Q.at<double>(2, 3);
    double Tx = -1.0 / Q.at<double>(3, 2);

    // Create key image (vertical color bar)
    int height = 480;
    int width = 200;
    cv::Mat key = cv::Mat::zeros(height, width, CV_8UC3);

    // Fill color bar
    for(int y = height - 1; y >= 0; y--) {
        // Normalized disparity from 0 to sbmSearchBound
        float disp = 1.0 - (float)(height - y - 1) / height;

        // Hue from 0 to 120 (red to green in HSV)
        uint8_t hue = (uint8_t)(120.0f * disp);

        // HSV to BGR
        cv::Mat hsv(1, 1, CV_8UC3);
        hsv.at<cv::Vec3b>(0,0) = cv::Vec3b(hue, 255, 255); // OpenCV HSV: H/2
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

        // Fill half of the row
        for(int x = 0; x < width/2; x++) {
            key.at<cv::Vec3b>(y, x) = bgr.at<cv::Vec3b>(0,0);
        }
    }

    // Add ticks and labels
    int numTicks = 5;
    for(int i = numTicks - 1; i >= 0; i--) {
        // Disparity from 2 to sbmSearchBound
        float disp = 10.0f + (float)i / (numTicks - 1) * (sbmSearchBound - 20);
        float depth = f * Tx / disp;

        int y = (int)((disp / sbmSearchBound) * height);
        if(y < 0) y = 0;
        if(y >= height) y = height - 1;

        // Draw tick mark
        cv::line(key, cv::Point(width - 10, y), cv::Point(width, y), cv::Scalar(255,255,255), 2);

        // Draw label
        std::string label = cv::format("%.2f m", depth);
        cv::putText(key, label, cv::Point(100, y+5), cv::FONT_HERSHEY_SIMPLEX,
                    0.4, cv::Scalar(255,255,255), 1);
    }

    // Add title
    //cv::putText(key, "Depth", cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX,
    //            0.6, cv::Scalar(255,255,255), 1);
    cv::putText(key, "Near", cv::Point(20, height-10), cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(200,200,200), 1);
    cv::putText(key, "Far", cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(200,200,200), 1);

    return key;
}

int main(int argc, char* argv[])
{

    namespace po = boost::program_options;

    string calibJSONPath;
    string hotPixelsDir;
    int highPassMicroseconds;
    double lowPassHz;
    int readCameraMilliseconds;
    int eventAccumulatorMilliseconds;
    double eventAccumulatorGain;
    int aaSurfacePatchesX;
    int aaSurfacePatchesY;
    double minBlockVariance;
    double minBlockCorrelation;
    int sbmEventDownsampling;
    int sbmHalfBlockWidth;
    int sbmHalfBlockHeight;
    int sbmNumThreads;
    int sbmSearchBound;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("calibration-json", po::value<std::string>(&calibJSONPath), "Camera calibration file written in JSON according to inivation standards")
        ("hot-pixels-dir", po::value<std::string>(&hotPixelsDir), "In which directory to find numpy files describing which camera pixels are hot")
        ("high-pass-us", po::value<int>(&highPassMicroseconds)->default_value(1000), "Background noise filter time constant in microseconds")
        ("low-pass-hz", po::value<double>(&lowPassHz)->default_value(500.0), "Low pass filter frequency (inverse of refractory period)")
        ("read-camera-ms", po::value<int>(&readCameraMilliseconds)->default_value(100), "Interval in milliseconds in which buffered data from the camera is sent to the rest of the system")
        ("event-accumulator-ms", po::value<int>(&eventAccumulatorMilliseconds)->default_value(1000), "Decay constant of the event accumulator in milliseconds")
        ("event-accumulator-gain", po::value<double>(&eventAccumulatorGain)->default_value(0.15), "Contribution of individual events to the accumulator")
        ("aa-patches-x", po::value<int>(&aaSurfacePatchesX)->default_value(4), "How many grid patches for adaptive accumulation, x axis")
        ("aa-patches-y", po::value<int>(&aaSurfacePatchesY)->default_value(3), "How many grid patches for adaptive accumulation, y axis")
        ("sbm-min-variance", po::value<double>(&minBlockVariance)->default_value(50.), "Minimum variance that a time surface patch must possess to be matched.")
        ("sbm-min-correlation", po::value<double>(&minBlockCorrelation)->default_value(.9), "Minimum correlation between two matched blocks in stereo block matching.")
        ("sbm-event-downsampling", po::value<int>(&sbmEventDownsampling)->default_value(100), "Rate at which we downsample events to get blocks for SBM")
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
    cout << "SBM downsampling rate of events to obtain block centers: " << sbmEventDownsampling << endl;
    cout << "SBM half block width: " << sbmHalfBlockWidth << endl;
    cout << "SBM half block height: " << sbmHalfBlockHeight << endl;
    cout << "SBM number of threads: " << sbmNumThreads << endl;
    cout << "SBM search bound: " << sbmSearchBound << endl;

    auto calibration = dv::camera::CalibrationSet::LoadFromFile(calibJSONPath);

    // Get calibration data
    auto leftCameraCalib = calibration.getCameraCalibration("C0").value();
    auto rightCameraCalib = calibration.getCameraCalibration("C1").value();
    const cv::Size resolution = leftCameraCalib.resolution;

    // Direct assignment - no element-by-element copying
    cv::Mat camMatL_mat = cv::Mat(leftCameraCalib.getCameraMatrix());  // CV_32F
    cv::Mat camMatR_mat = cv::Mat(rightCameraCalib.getCameraMatrix()); // CV_32F

    cv::Mat distCoeffsL_mat = cv::Mat(leftCameraCalib.distortion);     // CV_32F
    cv::Mat distCoeffsR_mat = cv::Mat(rightCameraCalib.distortion);    // CV_32F

    auto Reigen = rightCameraCalib.transformationToC0.getRotationMatrix();
    auto Teigen = rightCameraCalib.transformationToC0.getTranslation();
    cv::Mat R, T;
    cv::eigen2cv(Reigen, R);
    cv::eigen2cv(Teigen, T);

    // Convert everything to CV_64F for stereoRectify
    camMatL_mat.convertTo(camMatL_mat, CV_64F);
    camMatR_mat.convertTo(camMatR_mat, CV_64F);
    distCoeffsL_mat.convertTo(distCoeffsL_mat, CV_64F);
    distCoeffsR_mat.convertTo(distCoeffsR_mat, CV_64F);
    R.convertTo(R, CV_64F);
    T.convertTo(T, CV_64F);

    // Stereo rectify (CV_64F)
    cv::Mat R1, R2, P1, P2, Q;
    stereoRectify(camMatL_mat, distCoeffsL_mat, camMatR_mat, distCoeffsR_mat,
                resolution, R, T, R1, R2, P1, P2, Q);

    // Convert back to CV_32F for initUndistortRectifyMap
    cv::Mat camMatL_32, camMatR_32, distCoeffsL_32, distCoeffsR_32;
    cv::Mat R1_32, R2_32, P1_32, P2_32;

    camMatL_mat.convertTo(camMatL_32, CV_32F);
    camMatR_mat.convertTo(camMatR_32, CV_32F);
    distCoeffsL_mat.convertTo(distCoeffsL_32, CV_32F);
    distCoeffsR_mat.convertTo(distCoeffsR_32, CV_32F);
    R1.convertTo(R1_32, CV_32F);
    R2.convertTo(R2_32, CV_32F);
    P1.convertTo(P1_32, CV_32F);
    P2.convertTo(P2_32, CV_32F);

    // Generate maps (CV_32F)
    cv::Mat mapL1, mapL2, mapR1, mapR2;
    initUndistortRectifyMap(camMatL_32, distCoeffsL_32, R1_32, P1_32,
                            resolution, CV_32FC1, mapL1, mapL2);
    initUndistortRectifyMap(camMatR_32, distCoeffsR_32, R2_32, P2_32,
                            resolution, CV_32FC1, mapR1, mapR2);

    auto imuCalib = calibration.getImuCalibration("S0").value();
    cv::Point3f gyroBias = imuCalib.omegaOffsetAvg;
    cv::Point3f accelBias = imuCalib.accOffsetAvg;

    // Initialize a window to show previews of the output
    cv::namedWindow("Left", cv::WINDOW_NORMAL);
    cv::namedWindow("Right", cv::WINDOW_NORMAL);
    cv::namedWindow("Trajectory", cv::WINDOW_NORMAL);
    cv::namedWindow("Depth", cv::WINDOW_NORMAL);

    cv::Mat depthColorKey = drawDepthColorKey(ref(Q), sbmSearchBound);

    queue<dv::EventStore> leftEventsToMap;
    queue<vector<dv::IMU>> imuQueue;
    queue<cv::Mat> velocityVisQueue;
    queue<cv::Mat> leftImageToRender;
    queue<cv::Mat> rightImageToRender;
    queue<cv::Mat> leftImageToMap;
    queue<cv::Mat> rightImageToMap;
    queue<cv::Mat> depthImageQueue;

    cv::Mat trajectoryVisualization = cv::Mat::zeros(400, 400, CV_8UC1);

    thread leftCameraCaptureThread(
        &leftCameraCapture,
        resolution,
        leftCameraCalib.name,
        hotPixelsDir + "/hot_pixels_left_x.npy",
        hotPixelsDir + "/hot_pixels_left_y.npy",
        highPassMicroseconds,
        eventAccumulatorMilliseconds,
        eventAccumulatorGain,
        readCameraMilliseconds,
        //camMatL,
        //distCoeffsL,
        mapL1,
        mapL2,
        ref(leftEventsToMap),
        ref(leftImageToRender),
        ref(leftImageToMap),
        ref(imuQueue)
    );
    thread rightCameraCaptureThread(
        &rightCameraCapture,
        resolution,
        rightCameraCalib.name,
        hotPixelsDir + "/hot_pixels_right_x.npy",
        hotPixelsDir + "/hot_pixels_right_y.npy",
        highPassMicroseconds,
        eventAccumulatorMilliseconds,
        eventAccumulatorGain,
        readCameraMilliseconds,
        //camMatR,
        //distCoeffsR,
        mapR1,
        mapR2,
        ref(rightImageToRender),
        ref(rightImageToMap)
    );

    //thread trackingThread(
    //    &imuPreintegration,
    //    gyroBias,
    //    accelBias,
    //    ref(imuQueue),
    //    ref(velocityVisQueue)
    //);

    //thread leftImageRepresentationThread(
    //    &imageRepresentationLoop,
    //    resolution,
    //    eventAccumulatorMilliseconds,
    //    camMatL,
    //    distCoeffsL,
    //    ref(leftCameraToImage),
    //    ref(leftImageToRender),
    //    ref(leftImageToMap)
    //);
    //thread rightImageRepresentationThread(
    //    &imageRepresentationLoop,
    //    resolution,
    //    eventAccumulatorMilliseconds,
    //    camMatR,
    //    distCoeffsR,
    //    ref(rightCameraToImage),
    //    ref(rightImageToRender),
    //    ref(rightImageToMap)
    //);

    thread depthEstimationThread(
        &depthEstimationLoop,
        sbmNumThreads,
        minBlockVariance,
        minBlockCorrelation,
        resolution,
        sbmHalfBlockWidth,
        sbmHalfBlockHeight,
        sbmEventDownsampling,
        sbmSearchBound,
        ref(leftEventsToMap),
        ref(leftImageToMap),
        ref(rightImageToMap),
        ref(depthImageQueue)
    );

    // Run the processing loop while both cameras are connected
    while (true) {
        if (!leftImageToRender.empty() && !rightImageToRender.empty() && !depthImageQueue.empty()) {
            cv::Mat leftImage = leftImageToRender.front();
            leftImageToRender.pop();
            cv::Mat rightImage = rightImageToRender.front();
            rightImageToRender.pop();
            auto depthImage = depthImageQueue.front();
            depthImageQueue.pop();
            cv::Mat depthImageKey;
            cv::hconcat(depthImage, depthColorKey, depthImageKey);

            cv::imshow("Left", leftImage);
            cv::imshow("Right", rightImage);
            cv::imshow("Depth", depthImageKey);
        }
        //if (!velocityVisQueue.empty()) {
        //    cv::Mat trajectoryVisualization = velocityVisQueue.front();
        //    velocityVisQueue.pop();
        //    cv::imshow("Trajectory", trajectoryVisualization);
        //}
        if (!imuQueue.empty()) {
            imuQueue.pop();
        }
        // Wait for a small amount of time to avoid CPU overhaul
        cv::waitKey(2);
    }
    //*/
    return 0;
}