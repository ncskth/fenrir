#include <cameraCapture.h>

namespace SlamDemo {

    using namespace std::chrono_literals;
    using namespace std;

    barrier sync_point(2);

    void updateAccumulator(
        const cv::Size resolution,
        const double positiveFactor,
        const double negativeFactor,
        const dv::EventStore& events,
        vector<double>& accumulator
    ) {
        for(auto ev : events) {
            if(ev.polarity())  {
                accumulator[resolution.width*ev.y() + ev.x()] = min(1.0, positiveFactor*accumulator[resolution.width*ev.y() + ev.x()]);
            }
            else {
                accumulator[resolution.width*ev.y() + ev.x()] = max(0.0, negativeFactor*accumulator[resolution.width*ev.y() + ev.x()]);
            }
        }
        //cout << "Accumulator updated!" << endl;
    }

    cv::Mat generateImage(const cv::Size resolution, vector<double>& accumulator) {
        cv::Mat result(resolution, CV_8UC1);
        for(int i = 0; i < resolution.area(); i++) {
            result.at<uint8_t>(i/resolution.width, i%resolution.width) = (uint8_t)(255*accumulator[i]);
        }
        return result;
    }

    void rightCameraCapture(
        const cv::Size resolution,
        const string serial,
        const string hotPixelXFile,
        const string hotPixelYFile,
        const int highPassMicroseconds,
        const int accumulatorTimeConstant,
        const double accumulatorGain,
        const double positiveFactor,
        const double negativeFactor,
        const int sendIntervalMilliseconds,
        const cv::Matx33f cameraMatrix,
        const vector<float> distortionCoeffs,
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
        dv::noise::LowPassFilter lowPass(resolution, 1000.0);

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
        //dv::Accumulator accumulator(resolution);

        // Apply configuration, these values can be modified to taste
        //accumulator.setMinPotential(0.f);
        //accumulator.setMaxPotential(1.f);
        //accumulator.setNeutralPotential(0.5f);
        //accumulator.setEventContribution(accumulatorGain);
        //accumulator.setDecayFunction(dv::Accumulator::Decay::EXPONENTIAL);
        //accumulator.setDecayParam(1e3*accumulatorTimeConstant);
        //accumulator.setIgnorePolarity(false);
        //accumulator.setSynchronousDecay(false);

        vector<double> accumulator;
        accumulator.reserve(resolution.area());
        for(int i = 0; i < resolution.area(); i++) {
            accumulator[i] = 0.5;
        }

        cout << "Right camera ready!" << endl;
        sync_point.arrive_and_wait();

        auto lastQPush = chrono::high_resolution_clock::now();

        while (camera->isRunning()) {
            if (const auto raw = camera->getNextEventBatch()) {
                highPass.accept(*raw);
                const auto high = highPass.generateEvents();
                lowPass.accept(high);
                const auto low = lowPass.generateEvents();
                maskFilter.accept(low);
                const auto masked = maskFilter.generateEvents();
                updateAccumulator(resolution, positiveFactor, negativeFactor, ref(masked), ref(accumulator));
            }

            auto now = chrono::high_resolution_clock::now();
            if (now - lastQPush > sendIntervalMilliseconds * 1ms) {
                //dv::Frame frame = accumulator.generateFrame();
                //cv::Mat imageDistorted = frame.image;
                cv::Mat imageDistorted = generateImage(resolution, accumulator);
                cv::Mat image;
                cv::undistort(imageDistorted, image, cameraMatrix, distortionCoeffs);
                //cv::Mat blurred;
                //cv::blur(image, blurred, cv::Size(3, 3));
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
        const int accumulatorTimeConstant,
        const double accumulatorGain,
        const double positiveFactor,
        const double negativeFactor,
        const int sendIntervalMilliseconds,
        const cv::Matx33f cameraMatrix,
        const vector<float> distortionCoeffs,
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
        dv::noise::LowPassFilter lowPass(resolution, 1000.0);

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
        //dv::Accumulator accumulator(resolution);

        // Apply configuration, these values can be modified to taste
        //accumulator.setMinPotential(0.f);
        //accumulator.setMaxPotential(1.f);
        //accumulator.setNeutralPotential(0.5f);
        //accumulator.setEventContribution(accumulatorGain);
        //accumulator.setDecayFunction(dv::Accumulator::Decay::EXPONENTIAL);
        //accumulator.setDecayParam(1e3*accumulatorTimeConstant);
        //accumulator.setIgnorePolarity(false);
        //accumulator.setSynchronousDecay(false);

        vector<double> accumulator;
        accumulator.reserve(resolution.area());
        for(int i = 0; i < resolution.area(); i++) {
            accumulator[i] = 0.5;
        }

        cout << "Left camera ready!" << endl;
        sync_point.arrive_and_wait();

        auto lastQPush = chrono::high_resolution_clock::now();

        while (camera->isRunning()) {
            if (const auto raw = camera->getNextEventBatch()) {
                highPass.accept(*raw);
                const auto high = highPass.generateEvents();
                lowPass.accept(high);
                const auto low = lowPass.generateEvents();
                maskFilter.accept(low);
                const auto masked = maskFilter.generateEvents();
                updateAccumulator(resolution, positiveFactor, negativeFactor, ref(masked), ref(accumulator));
                eventBuffer.add(*raw);
            }

            if (const auto imuBatch = camera->getNextImuBatch()) {
                imuBuffer.insert(imuBuffer.end(), imuBatch->begin(), imuBatch->end());
            }

            auto now = chrono::high_resolution_clock::now();
            if (now - lastQPush > sendIntervalMilliseconds * 1ms && imuBuffer.size() > 1 && eventBuffer.size() > 1) {
                outgoingEvents.push(eventBuffer);
                eventBuffer = dv::EventStore();

                //dv::Frame frame = accumulator.generateFrame();
                //cv::Mat imageDistorted = frame.image;
                cv::Mat imageDistorted = generateImage(resolution, accumulator);
                cv::Mat image;
                cv::undistort(imageDistorted, image, cameraMatrix, distortionCoeffs);
                //cv::Mat blurred;
                //cv::blur(image, blurred, cv::Size(3, 3));
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