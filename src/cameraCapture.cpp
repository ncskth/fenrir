#include <cameraCapture.h>

namespace SlamDemo {

    using namespace std::chrono_literals;
    using namespace std;

    barrier sync_point(2);

    void updateImageAndTimestamps(
        const double decay,
        const double gain,
        const int width,
        const cv::Mat undistortRectifyMap1,
        const cv::Mat undistortRectifyMap2,
        const dv::EventStore& events,
        cv::Mat& image,
        vector<int64_t>& timestamps
    ) {
        for(dv::Event ev : events) {
            int x = undistortRectifyMap1.at<int64_t>(ev.x(), ev.y());
            int y = undistortRectifyMap2.at<int64_t>(ev.x(), ev.y());
            double diff = ev.polarity() ? gain : -gain;
            double decayed = exp(( timestamps[width*y + x] - ev.timestamp() )/decay) * image.at<double>(x, y);
            image.at<double>(x, y) = decayed + diff;
            timestamps[width*y + x] = ev.timestamp();
        }
    }

    void rightCameraCapture(
        const cv::Size resolution,
        const string serial,
        const string hotPixelXFile,
        const string hotPixelYFile,
        const int highPassMicroseconds,
        const int accumulatorTimeConstant,
        const double accumulatorGain,
        const int sendIntervalMilliseconds,
        const cv::Mat undistortRectifyMat1,
        const cv::Mat undistortRectifyMat2,
        cv::Mat& image,
        vector<int64_t>& timestamps
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
        dv::EventMaskFilter maskFilter(mask);

        //dv::EventStore eventBuffer;

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
        //accumulator.setSynchronousDecay(false);

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
                updateImageAndTimestamps(
                    accumulatorTimeConstant,
                    accumulatorGain,
                    resolution.width,
                    undistortRectifyMat1,
                    undistortRectifyMat2,
                    masked,
                    image,
                    timestamps);
            }

            /*
            auto now = chrono::high_resolution_clock::now();
            if (now - lastQPush > sendIntervalMilliseconds * 1ms) {
                dv::Frame frame = accumulator.generateFrame();
                cv::Mat imageDistorted = frame.image;
                cv::Mat image;
                //cv::undistort(imageDistorted, image, cameraMatrix, distortionCoeffs);
                cv::remap(imageDistorted, image, undistortRectifyMat1, undistortRectifyMat2, cv::INTER_LINEAR);
                cv::Mat blurred;
                cv::blur(image, blurred, cv::Size(3, 3));
                outgoingImages1.push(blurred);
                cv::Mat frameToDepth;
                blurred.convertTo(frameToDepth, CV_64F);
                outgoingImages2.push(frameToDepth);

                sync_point.arrive_and_wait();
                lastQPush = now;
            }
            */
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
        const int sendIntervalMilliseconds,
        const int maxEventsInBuffer,
        const cv::Mat undistortRectifyMat1,
        const cv::Mat undistortRectifyMat2,
        queue<vector<dv::Event>>& outgoingEvents,
        cv::Mat& image,
        vector<int64_t>& timestamps,
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
        dv::EventMaskFilter maskFilter(mask);

        vector<dv::Event> eventBuffer(maxEventsInBuffer);
        int bufferIndex = 0;
        vector<dv::IMU> imuBuffer;

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
        //accumulator.setSynchronousDecay(false);

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
                updateImageAndTimestamps(
                    accumulatorTimeConstant,
                    accumulatorGain,
                    resolution.width,
                    undistortRectifyMat1,
                    undistortRectifyMat2,
                    masked,
                    image,
                    timestamps);
                //eventBuffer.add(masked);
                //accumulator.accept(masked);
                for(dv::Event ev : masked) {
                    eventBuffer[bufferIndex] = ev;
                    bufferIndex++;
                    bufferIndex %= maxEventsInBuffer;
                }
            }

            if (const auto imuBatch = camera->getNextImuBatch()) {
                imuBuffer.insert(imuBuffer.end(), imuBatch->begin(), imuBatch->end());
            }

            auto now = chrono::high_resolution_clock::now();
            if (now - lastQPush > sendIntervalMilliseconds * 1ms && imuBuffer.size() > 1 && eventBuffer.size() > 1) {
                outgoingEvents.push(eventBuffer);
                //eventBuffer = dv::EventStore();
                bufferIndex = 0;

                //dv::Frame frame = accumulator.generateFrame();
                //cv::Mat imageDistorted = frame.image;
                //cv::Mat image;
                ////cv::undistort(imageDistorted, image, cameraMatrix, distortionCoeffs);
                //cv::remap(imageDistorted, image, undistortRectifyMat1, undistortRectifyMat2, cv::INTER_LINEAR);
                //cv::Mat blurred;
                //cv::blur(image, blurred, cv::Size(3, 3));
                //outgoingImages1.push(blurred);
                //cv::Mat frameToDepth;
                //blurred.convertTo(frameToDepth, CV_64F);
                //outgoingImages2.push(frameToDepth);
                //outgoingIMU.push(imuBuffer);
                //imuBuffer.clear();

                //sync_point.arrive_and_wait();
                //lastQPush = now;
            }
        }
    }
}