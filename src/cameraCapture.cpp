#include <cameraCapture.h>

namespace SlamDemo {

    using namespace std::chrono_literals;
    using namespace std;

    barrier sync_point(2);

    tuple<cv::Mat, cv::Mat> buildScatterIndex(
        const cv::Mat& undistortRectifyMap1,
        const cv::Mat& undistortRectifyMap2,
        const int height, const int width
    ) {
        cv::Mat mapX(height, width, CV_32SC1);
        cv::Mat mapY(height, width, CV_32SC1);
        cv::Mat scoreX = cv::Mat::ones(height, width, CV_32FC1);
        cv::Mat scoreY = cv::Mat::ones(height, width, CV_32FC1);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float xf = undistortRectifyMap1.at<float>(y, x);
                float yf = undistortRectifyMap2.at<float>(y, x);

                long xl = lround(xf);
                long yl = lround(yf);

                float errx = abs(xf - xl);
                float erry = abs(yf - yl);

                // first check if the nearest integer pixel is the best fit
                //if(errx < scoreX.at<float>(yl, xl) && erry < scoreY.at<float>(yl, xl)) {
                    mapX.at<int>(yl, xl) = x;
                    mapY.at<int>(yl, xl) = y;
                    scoreX.at<float>(yl, xl) = errx;
                    scoreY.at<float>(yl, xl) = erry;
                //}
                /*
                // otherwise check all four neighbors
                else {
                    // technically this is not the most efficient logic, but it doesn't matter
                    xl = floorl(xf);
                    yl = floorl(yf);
                    errx = abs(xf - xl);
                    erry = abs(yf - yl);
                    if(errx < scoreX.at<float>(yl, xl) && erry < scoreY.at<float>(yl, xl)) {
                        mapX.at<int>(yl, xl) = x;
                        mapY.at<int>(yl, xl) = y;
                        scoreX.at<float>(yl, xl) = errx;
                        scoreY.at<float>(yl, xl) = erry;
                    }
                    else {
                        xl++;
                        errx = abs(xf - xl);
                        if(errx < scoreX.at<float>(yl, xl) && erry < scoreY.at<float>(yl, xl)) {
                            mapX.at<int>(yl, xl) = x;
                            mapY.at<int>(yl, xl) = y;
                            scoreX.at<float>(yl, xl) = errx;
                            scoreY.at<float>(yl, xl) = erry;
                        }
                        else {
                            yl++;
                            erry = abs(yf - yl);
                            if(errx < scoreX.at<float>(yl, xl) && erry < scoreY.at<float>(yl, xl)) {
                                mapX.at<int>(yl, xl) = x;
                                mapY.at<int>(yl, xl) = y;
                                scoreX.at<float>(yl, xl) = errx;
                                scoreY.at<float>(yl, xl) = erry;
                            }
                            else {
                                xl--;
                                errx = abs(xf - xl);
                                //if(errx < scoreX.at<float>(yl, xl) && erry < scoreY.at<float>(yl, xl)) {
                                    mapX.at<int>(yl, xl) = x;
                                    mapY.at<int>(yl, xl) = y;
                                    scoreX.at<float>(yl, xl) = errx;
                                    scoreY.at<float>(yl, xl) = erry;
                                //}
                            }
                        }
                    }
                }
                */
            }
        }

        /*
        // re-scan the whole array to re-assign pixels that lost their place
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float xf = undistortRectifyMap1.at<float>(y, x);
                float yf = undistortRectifyMap2.at<float>(y, x);

                long xl = lround(xf);
                long yl = lround(yf);

                float errx = abs(xf - xl);
                float erry = abs(yf - yl);

                // first check if the nearest integer pixel is the best fit
                if(errx < scoreX.at<float>(yl, xl) && erry < scoreY.at<float>(yl, xl)) {
                    mapX.at<int>(yl, xl) = x;
                    mapY.at<int>(yl, xl) = y;
                    scoreX.at<float>(yl, xl) = errx;
                    scoreY.at<float>(yl, xl) = erry;
                }
                // otherwise check all four neighbors
                else {
                    // technically this is not the most efficient logic, but it doesn't matter
                    xl = floorl(xf);
                    yl = floorl(yf);
                    errx = abs(xf - xl);
                    erry = abs(yf - yl);
                    if(errx < scoreX.at<float>(yl, xl) && erry < scoreY.at<float>(yl, xl)) {
                        mapX.at<int>(yl, xl) = x;
                        mapY.at<int>(yl, xl) = y;
                        scoreX.at<float>(yl, xl) = errx;
                        scoreY.at<float>(yl, xl) = erry;
                    }
                    else {
                        xl++;
                        errx = abs(xf - xl);
                        if(errx < scoreX.at<float>(yl, xl) && erry < scoreY.at<float>(yl, xl)) {
                            mapX.at<int>(yl, xl) = x;
                            mapY.at<int>(yl, xl) = y;
                            scoreX.at<float>(yl, xl) = errx;
                            scoreY.at<float>(yl, xl) = erry;
                        }
                        else {
                            yl++;
                            erry = abs(yf - yl);
                            if(errx < scoreX.at<float>(yl, xl) && erry < scoreY.at<float>(yl, xl)) {
                                mapX.at<int>(yl, xl) = x;
                                mapY.at<int>(yl, xl) = y;
                                scoreX.at<float>(yl, xl) = errx;
                                scoreY.at<float>(yl, xl) = erry;
                            }
                            else {
                                xl--;
                                errx = abs(xf - xl);
                                if(errx < scoreX.at<float>(yl, xl) && erry < scoreY.at<float>(yl, xl)) {
                                    mapX.at<int>(yl, xl) = x;
                                    mapY.at<int>(yl, xl) = y;
                                    scoreX.at<float>(yl, xl) = errx;
                                    scoreY.at<float>(yl, xl) = erry;
                                }
                            }
                        }
                    }
                }
            }
        }
        */

        return {mapX, mapY};
    }

    void updateImageAndTimestamps(
        const double decay,
        const double gain,
        const int height,
        const int width,
        const cv::Mat& undistortRectifyMap1,
        const cv::Mat& undistortRectifyMap2,
        const dv::EventStore& events,
        cv::Mat& image,
        vector<int64_t>& timestamps
    ) {
        for(dv::Event ev : events) {
            // roughly 90-100 ops per event
            int x = undistortRectifyMap1.at<int>(ev.y(), ev.x());
            int y = undistortRectifyMap2.at<int>(ev.y(), ev.x());
            double diff = ev.polarity() ? gain : -gain;

            for(int px = max(0, x - 1); px < min(width, x + 2); px++) {
                for(int py = max(0, y - 1); py < min(height, y + 2); py++) {
                    double val = image.at<double>(py, px);

                    if((ev.polarity() && val < 0.9) || (!ev.polarity() && val > 0.1)) {
                        // timestamps are in microseconds, decay is in milliseconds
                        double decayed = exp(1e-3*( timestamps[width*py + px] - ev.timestamp() )/decay) * (val - 0.5);
                        image.at<double>(py, px) = decayed + 0.111111*diff + 0.5;
                        timestamps[width*py + px] = ev.timestamp();
                    }
                }
            }
        }
    }

    void rightCameraCapture(
        const cv::Size resolution,
        const string serial,
        const string hotPixelXFile,
        const string hotPixelYFile,
        const double lowPassHz,
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
        //accumulator.setMinPotential(0.f);
        //accumulator.setMaxPotential(1.f);
        //accumulator.setNeutralPotential(0.5f);
        //accumulator.setEventContribution(accumulatorGain);
        //accumulator.setDecayFunction(dv::Accumulator::Decay::EXPONENTIAL);
        //accumulator.setDecayParam(1e3*accumulatorTimeConstant);
        //accumulator.setIgnorePolarity(false);
        //accumulator.setSynchronousDecay(false);

        auto inverted = buildScatterIndex(undistortRectifyMat1, undistortRectifyMat2, resolution.height, resolution.width);
        cv::Mat redistortRectifyMat1 = get<0>(inverted);
        cv::Mat redistortRectifyMat2 = get<1>(inverted);

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
                updateImageAndTimestamps(
                    accumulatorTimeConstant,
                    accumulatorGain,
                    resolution.height,
                    resolution.width,
                    ref(redistortRectifyMat1),
                    ref(redistortRectifyMat2),
                    masked,
                    image,
                    timestamps);
            }

            auto now = chrono::high_resolution_clock::now();
            if (now - lastQPush > sendIntervalMilliseconds * 1ms) {
                /*
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
                */

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
        const double lowPassHz,
        const int highPassMicroseconds,
        const int accumulatorTimeConstant,
        const double accumulatorGain,
        const int sendIntervalMilliseconds,
        const int maxEventsInBuffer,
        const cv::Mat undistortRectifyMat1,
        const cv::Mat undistortRectifyMat2,
        queue<vector<tuple<int, int>>>& outgoingEvents,
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

        vector<tuple<int, int>> eventBuffer(maxEventsInBuffer);
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

        auto inverted = buildScatterIndex(undistortRectifyMat1, undistortRectifyMat2, resolution.height, resolution.width);
        cv::Mat redistortRectifyMat1 = get<0>(inverted);
        cv::Mat redistortRectifyMat2 = get<1>(inverted);

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
                updateImageAndTimestamps(
                    accumulatorTimeConstant,
                    accumulatorGain,
                    resolution.height,
                    resolution.width,
                    ref(redistortRectifyMat1),
                    ref(redistortRectifyMat2),
                    masked,
                    image,
                    timestamps);
                //eventBuffer.add(masked);
                //accumulator.accept(masked);
                for(dv::Event ev : *raw) {
                    //cout << "Trying to append to event buffer" << endl;
                    eventBuffer[bufferIndex] = {(int)ev.x(), (int)ev.y()};
                    //cout << "Appended to event buffer" << endl;
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

                sync_point.arrive_and_wait();
                lastQPush = now;
            }
        }
    }
}