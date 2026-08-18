#include <cameraCapture.h>

namespace SlamDemo {

    using namespace std::chrono_literals;
    using namespace std;

    barrier sync_point(2);

    void rightCameraCapture(
        const cv::Size resolution,
        const string serial,
        const string hotPixelXFile,
        const string hotPixelYFile,
        const int highPassMicroseconds,
        //const int accumulatorTimeConstant,
        //const double accumulatorGain,
        const int sendIntervalMilliseconds,
        //const cv::Matx33f cameraMatrix,
        //const vector<float> distortionCoeffs,
        queue<dv::EventStore>& outgoingEvents
        ) {

        // Open the stereo camera with camera names from calibration
        auto camera  = dv::io::camera::open(serial);

        // Make sure both cameras support event stream output, throw an error otherwise
        if (!camera->isEventStreamAvailable()) {
            throw dv::exceptions::RuntimeError("Input camera does not provide an event stream.");
        }

        dv::noise::BackgroundActivityNoiseFilter highPass(resolution, highPassMicroseconds*1us);
        //dv::noise::LowPassFilter lowPass(resolution, lowPassHz);

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

        // Initialize an accumulator with some resolution
        dv::Accumulator accumulator(resolution);

        // Apply configuration, these values can be modified to taste
        //accumulator.setMinPotential(0.f);
        //accumulator.setMaxPotential(1.f);
        //accumulator.setNeutralPotential(0.5f);
        //accumulator.setEventContribution(accumulatorGain);
        //accumulator.setDecayFunction(dv::Accumulator::Decay::EXPONENTIAL);
        //accumulator.setDecayParam(1e3*accumulatorTimeConstant);
        //accumulator.setIgnorePolarity(false);
        //accumulator.setSynchronousDecay(true);

        cout << "Right camera ready!" << endl;
        sync_point.arrive_and_wait();

        auto lastQPush = chrono::high_resolution_clock::now();

        while (camera->isRunning()) {
            if (const auto raw = camera->getNextEventBatch()) {
                highPass.accept(*raw);
                const auto high = highPass.generateEvents();
                //lowPass.accept(high);
                //const auto low = lowPass.generateEvents();
                maskFilter.accept(high);
                const auto masked = maskFilter.generateEvents();
                eventBuffer.add(masked);
                //accumulator.accept(masked);
            }

            auto now = chrono::high_resolution_clock::now();
            if (now - lastQPush > sendIntervalMilliseconds * 1ms) {
                outgoingEvents.push(eventBuffer);
                eventBuffer = dv::EventStore();

                //dv::Frame frame = accumulator.generateFrame();
                //cv::Mat imageDistorted = frame.image;
                //cv::Mat image;
                //cv::undistort(imageDistorted, image, cameraMatrix, distortionCoeffs);
                //cv::Mat blurred;
                //cv::blur(image, blurred, cv::Size(5, 5));
                //outgoingImages1.push(image);
                //cv::Mat frameToDepth;
                //image.convertTo(frameToDepth, CV_64F);
                //outgoingImages2.push(frameToDepth);

                sync_point.arrive_and_wait();
                lastQPush = now;
            }
        }
    }

    void leftCameraCapture(
        const cv::Size resolution,
        const string serial,
        const string hotPixelXFile,
        const string hotPixelYFile,
        const int highPassMicroseconds,
        //const int accumulatorTimeConstant,
        //const double accumulatorGain,
        const int sendIntervalMilliseconds,
        //const cv::Matx33f cameraMatrix,
        //const vector<float> distortionCoeffs,
        //queue<dv::EventStore>& outgoingEvents,
        queue<dv::EventStore>& outgoingEvents1,
        queue<dv::EventStore>& outgoingEvents2,
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
        //dv::noise::LowPassFilter lowPass(resolution, lowPassHz);

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

        // Initialize an accumulator with some resolution
        //dv::Accumulator accumulator(resolution);

        // Apply configuration, these values can be modified to taste
        //accumulator.setMinPotential(0.f);
        //accumulator.setMaxPotential(1.f);
        //accumulator.setNeutralPotential(0.5f);
        //accumulator.setEventContribution(accumulatorGain);
        //accumulator.setDecayFunction(dv::Accumulator::Decay::EXPONENTIAL);
        //accumulator.setDecayParam(1e3*accumulatorTimeConstant);
        //accumulator.setIgnorePolarity(false);
        //accumulator.setSynchronousDecay(true);

        cout << "Left camera ready!" << endl;
        sync_point.arrive_and_wait();

        auto lastQPush = chrono::high_resolution_clock::now();

        while (camera->isRunning()) {
            if (const auto raw = camera->getNextEventBatch()) {
                highPass.accept(*raw);
                const auto high = highPass.generateEvents();
                //lowPass.accept(high);
                //const auto low = lowPass.generateEvents();
                maskFilter.accept(high);
                const auto masked = maskFilter.generateEvents();
                eventBuffer.add(masked);
                //accumulator.accept(masked);
            }

            if (const auto imuBatch = camera->getNextImuBatch()) {
                imuBuffer.insert(imuBuffer.end(), imuBatch->begin(), imuBatch->end());
            }

            auto now = chrono::high_resolution_clock::now();
            if (now - lastQPush > sendIntervalMilliseconds * 1ms && imuBuffer.size() > 1 && eventBuffer.size() > 1) {
                outgoingEvents1.push(eventBuffer);
                outgoingEvents2.push(eventBuffer);
                eventBuffer = dv::EventStore();

                //dv::Frame frame = accumulator.generateFrame();
                //cv::Mat imageDistorted = frame.image;
                //cv::Mat image;
                //cv::undistort(imageDistorted, image, cameraMatrix, distortionCoeffs);
                //cv::Mat blurred;
                //cv::blur(image, blurred, cv::Size(5, 5));
                //outgoingImages1.push(image);
                //cv::Mat frameToDepth;
                //image.convertTo(frameToDepth, CV_64F);
                //outgoingImages2.push(frameToDepth);

                outgoingIMU.push(imuBuffer);
                imuBuffer.clear();

                sync_point.arrive_and_wait();
                lastQPush = now;
            }
        }
    }
}