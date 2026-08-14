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
        const double lowPassHz,
        const int sendIntervalMilliseconds,
        queue<cv::Mat>& outgoingImages1,
        queue<cv::Mat>& outgoingImages2
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
        dv::EventMaskFilter maskFilter(mask);

        //dv::EventStore eventBuffer;

        // Initialize an accumulator with some resolution
        dv::Accumulator accumulator(resolution);

        // Apply configuration, these values can be modified to taste
        accumulator.setMinPotential(0.f);
        accumulator.setMaxPotential(1.f);
        accumulator.setNeutralPotential(0.5f);
        accumulator.setEventContribution(0.15f);
        accumulator.setDecayFunction(dv::Accumulator::Decay::EXPONENTIAL);
        accumulator.setDecayParam(1e+6);
        accumulator.setIgnorePolarity(false);
        accumulator.setSynchronousDecay(false);

        cout << "Right camera ready!" << endl;
        sync_point.arrive_and_wait();

        auto lastQPush = chrono::high_resolution_clock::now();

        while (camera->isRunning()) {
            if (const auto raw = camera->getNextEventBatch()) {
                //highPass.accept(*raw);
                //const auto high = highPass.generateEvents();
                //lowPass.accept(high);
                //const auto low = lowPass.generateEvents();
                maskFilter.accept(*raw);
                const auto masked = maskFilter.generateEvents();
                //eventBuffer.add(masked);
                accumulator.accept(masked);
            }

            auto now = chrono::high_resolution_clock::now();
            if (now - lastQPush > sendIntervalMilliseconds * 1ms) {
                dv::Frame frame = accumulator.generateFrame();
                cv::Mat image = frame.image;
                outgoingImages1.push(image);
                cv::Mat frameToDepth;
                image.convertTo(frameToDepth, CV_64F);
                outgoingImages2.push(frameToDepth);

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
        const double lowPassHz,
        const int sendIntervalMilliseconds,
        queue<dv::EventStore>& outgoingEvents,
        queue<cv::Mat>& outgoingImages1,
        queue<cv::Mat>& outgoingImages2,
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
        dv::EventMaskFilter maskFilter(mask);

        dv::EventStore eventBuffer;
        vector<dv::IMU> imuBuffer;

        // Initialize an accumulator with some resolution
        dv::Accumulator accumulator(resolution);

        // Apply configuration, these values can be modified to taste
        accumulator.setMinPotential(0.f);
        accumulator.setMaxPotential(1.f);
        accumulator.setNeutralPotential(0.5f);
        accumulator.setEventContribution(0.15f);
        accumulator.setDecayFunction(dv::Accumulator::Decay::EXPONENTIAL);
        accumulator.setDecayParam(1e+6);
        accumulator.setIgnorePolarity(false);
        accumulator.setSynchronousDecay(false);

        cout << "Left camera ready!" << endl;
        sync_point.arrive_and_wait();

        auto lastQPush = chrono::high_resolution_clock::now();

        while (camera->isRunning()) {
            if (const auto raw = camera->getNextEventBatch()) {
                //highPass.accept(*raw);
                //const auto high = highPass.generateEvents();
                //lowPass.accept(high);
                //const auto low = lowPass.generateEvents();
                maskFilter.accept(*raw);
                const auto masked = maskFilter.generateEvents();
                eventBuffer.add(masked);
                accumulator.accept(masked);
            }

            if (const auto imuBatch = camera->getNextImuBatch()) {
                imuBuffer.insert(imuBuffer.end(), imuBatch->begin(), imuBatch->end());
            }

            auto now = chrono::high_resolution_clock::now();
            if (now - lastQPush > sendIntervalMilliseconds * 1ms && imuBuffer.size() > 1 && eventBuffer.size() > 1) {
                outgoingEvents.push(eventBuffer);
                eventBuffer = dv::EventStore();

                dv::Frame frame = accumulator.generateFrame();
                cv::Mat image = frame.image;
                outgoingImages1.push(image);
                cv::Mat frameToDepth;
                image.convertTo(frameToDepth, CV_64F);
                outgoingImages2.push(frameToDepth);

                outgoingIMU.push(imuBuffer);
                imuBuffer.clear();

                sync_point.arrive_and_wait();
                lastQPush = now;
            }
        }
    }
}