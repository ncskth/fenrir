#include <cameraCapture.h>

namespace SlamDemo {

    using namespace std::chrono_literals;
    using namespace std;

    barrier sync_point(2);

    ReverseUndistortRectifyMap buildInterpolationMap(
        const cv::Mat& undistortRectifyMap1,
        const cv::Mat& undistortRectifyMap2,
        const int height, const int width
    ) {
        cv::Mat topLeftToX(height, width, CV_32SC1);
        cv::Mat topLeftToY(height, width, CV_32SC1);
        cv::Mat topRightToX(height, width, CV_32SC1);
        cv::Mat topRightToY(height, width, CV_32SC1);
        cv::Mat bottomLeftToX(height, width, CV_32SC1);
        cv::Mat bottomLeftToY(height, width, CV_32SC1);
        cv::Mat bottomRightToX(height, width, CV_32SC1);
        cv::Mat bottomRightToY(height, width, CV_32SC1);

        cv::Mat topLeftToXWeight(height, width, CV_32FC1);
        cv::Mat topLeftToYWeight(height, width, CV_32FC1);
        cv::Mat topRightToXWeight(height, width, CV_32FC1);
        cv::Mat topRightToYWeight(height, width, CV_32FC1);
        cv::Mat bottomLeftToXWeight(height, width, CV_32FC1);
        cv::Mat bottomLeftToYWeight(height, width, CV_32FC1);
        cv::Mat bottomRightToXWeight(height, width, CV_32FC1);
        cv::Mat bottomRightToYWeight(height, width, CV_32FC1);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float xf = undistortRectifyMap1.at<float>(y, x);
                float yf = undistortRectifyMap2.at<float>(y, x);

                int xi = max(0, (int)floorl(xf));
                int yi = max(0, (int)floorl(yf));

                float wHoriz = xf - xi;
                float wVert = yf - yi;

                topLeftToX.at<int>(min(height, yi + 1), xi) = x;
                topLeftToY.at<int>(min(height, yi + 1), xi) = y;
                topRightToX.at<int>(min(height, yi + 1), min(width, xi + 1)) = x;
                topRightToY.at<int>(min(height, yi + 1), min(width, xi + 1)) = y;
                bottomLeftToX.at<int>(yi, xi) = x;
                bottomLeftToY.at<int>(yi, xi) = y;
                bottomRightToX.at<int>(yi, min(width, xi + 1)) = x;
                bottomRightToY.at<int>(yi, min(width, xi + 1)) = y;

                topLeftToXWeight.at<float>(min(height, yi + 1), xi) = 1.0 - wHoriz;
                topLeftToYWeight.at<float>(min(height, yi + 1), xi) = wVert;
                topRightToXWeight.at<float>(min(height, yi + 1), min(width, xi + 1)) = wHoriz;
                topRightToYWeight.at<float>(min(height, yi + 1), min(width, xi + 1)) = wVert;
                bottomLeftToXWeight.at<float>(yi, xi) = 1.0 - wHoriz;
                bottomLeftToYWeight.at<float>(yi, xi) = 1.0 - wVert;
                bottomRightToXWeight.at<float>(yi, min(width, xi + 1)) = 1.0 - wHoriz;
                bottomRightToYWeight.at<float>(yi, min(width, xi + 1)) = wVert;
            }
        }

        return ReverseUndistortRectifyMap{
            topLeftToX,
            topLeftToY,
            topRightToX,
            topRightToY,
            bottomLeftToX,
            bottomLeftToY,
            bottomRightToX,
            bottomRightToY,
            topLeftToXWeight,
            topLeftToYWeight,
            topRightToXWeight,
            topRightToYWeight,
            bottomLeftToXWeight,
            bottomLeftToYWeight,
            bottomRightToXWeight,
            bottomRightToYWeight
        };
    }

    void updateImageAndTimestamps(
        const double decay,
        const double gain,
        const int height,
        const int width,
        const ReverseUndistortRectifyMap& interpolationMap,
        const dv::EventStore& events,
        cv::Mat& image,
        vector<int64_t>& timestamps
    ) {
        for(dv::Event ev : events) {

            // BOTTOM LEFT

            int xi = max(0, (int)floorl(ev.x()));
            int yi = max(0, (int)floorl(ev.y()));

            int x = interpolationMap.bottomLeftToX.at<int>(yi, xi);
            int y = interpolationMap.bottomLeftToY.at<int>(yi, xi);
            double val = image.at<double>(y, x);

            if((ev.polarity() && val < 0.99) || (!ev.polarity() && val > 0.01)) {
                float weight = interpolationMap.bottomLeftToXWeight.at<float>(yi, xi)
                    * interpolationMap.bottomLeftToYWeight.at<float>(yi, xi);

                double diff = ev.polarity() ? weight*gain : -weight*gain;
                // timestamps are in microseconds, decay is in milliseconds
                double decayed = exp(1e-3*( timestamps[width*y + x] - ev.timestamp() )/decay) * (val - 0.5);
                image.at<double>(y, x) = decayed + diff + 0.5;
                timestamps[width*y + x] = ev.timestamp();
            }

            //cout << "Bottom left interpolation worked!" << endl;

            // BOTTOM RIGHT

            x = interpolationMap.bottomRightToX.at<int>(yi, min(width, xi + 1));
            y = interpolationMap.bottomRightToY.at<int>(yi, min(width, xi + 1));
            val = image.at<double>(y, x);

            if((ev.polarity() && val < 0.99) || (!ev.polarity() && val > 0.01)) {
                float weight = interpolationMap.bottomRightToXWeight.at<float>(yi, min(width, xi + 1))
                    * interpolationMap.bottomRightToYWeight.at<float>(yi, min(width, xi + 1));

                double diff = ev.polarity() ? weight*gain : -weight*gain;
                // timestamps are in microseconds, decay is in milliseconds
                double decayed = exp(1e-3*( timestamps[width*y + x] - ev.timestamp() )/decay) * (val - 0.5);
                image.at<double>(y, x) = decayed + diff + 0.5;
                timestamps[width*y + x] = ev.timestamp();
            }

            //cout << "Bottom right interpolation worked!" << endl;

            // TOP RIGHT

            x = interpolationMap.topRightToX.at<int>(min(height, yi + 1), min(width, xi + 1));
            y = interpolationMap.topRightToY.at<int>(min(height, yi + 1), min(width, xi + 1));
            val = image.at<double>(y, x);

            if((ev.polarity() && val < 0.99) || (!ev.polarity() && val > 0.01)) {
                float weight = interpolationMap.topRightToXWeight.at<float>(min(height, yi + 1), min(width, xi + 1))
                    * interpolationMap.topRightToYWeight.at<float>(min(height, yi + 1), min(width, xi + 1));

                double diff = ev.polarity() ? weight*gain : -weight*gain;
                // timestamps are in microseconds, decay is in milliseconds
                double decayed = exp(1e-3*( timestamps[width*y + x] - ev.timestamp() )/decay) * (val - 0.5);
                image.at<double>(y, x) = decayed + diff + 0.5;
                timestamps[width*y + x] = ev.timestamp();
            }

            //cout << "Top right interpolation worked!" << endl;

            // TOP LEFT

            x = interpolationMap.topLeftToX.at<int>(min(height, yi + 1), xi);
            y = interpolationMap.topLeftToY.at<int>(min(height, yi + 1), xi);
            val = image.at<double>(y, x);

            if((ev.polarity() && val < 0.99) || (!ev.polarity() && val > 0.01)) {
                float weight = interpolationMap.topLeftToXWeight.at<float>(min(height, yi + 1), xi)
                    * interpolationMap.topLeftToYWeight.at<float>(min(height, yi + 1), xi);

                double diff = ev.polarity() ? weight*gain : -weight*gain;
                // timestamps are in microseconds, decay is in milliseconds
                double decayed = exp(1e-3*( timestamps[width*y + x] - ev.timestamp() )/decay) * (val - 0.5);
                image.at<double>(y, x) = decayed + diff + 0.5;
                timestamps[width*y + x] = ev.timestamp();
            }

            //cout << "Top left interpolation worked!" << endl;
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

        ReverseUndistortRectifyMap interpolationMap = buildInterpolationMap(
            undistortRectifyMat1, undistortRectifyMat2, resolution.height, resolution.width);

        cout << "Right camera ready!" << endl;
        sync_point.arrive_and_wait();

        auto lastQPush = chrono::high_resolution_clock::now();

        while (camera->isRunning()) {
            if (const auto raw = camera->getNextEventBatch()) {
                highPass.accept(*raw);
                const auto high = highPass.generateEvents();
                //lowPass.accept(high);
                //const auto low = lowPass.generateEvents();
                //maskFilter.accept(*raw);
                //const auto masked = maskFilter.generateEvents();
                updateImageAndTimestamps(
                    accumulatorTimeConstant,
                    accumulatorGain,
                    resolution.height,
                    resolution.width,
                    ref(interpolationMap),
                    high,
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

        vector<tuple<int, int>> eventBuffer(maxEventsInBuffer);
        int bufferIndex = 0;
        vector<dv::IMU> imuBuffer;

        // Initialize an accumulator with some resolution
        dv::Accumulator accumulator(resolution);

        ReverseUndistortRectifyMap interpolationMap = buildInterpolationMap(
            undistortRectifyMat1, undistortRectifyMat2, resolution.height, resolution.width);

        cout << "Left camera ready!" << endl;
        sync_point.arrive_and_wait();

        auto lastQPush = chrono::high_resolution_clock::now();

        while (camera->isRunning()) {
            if (const auto raw = camera->getNextEventBatch()) {
                highPass.accept(*raw);
                const auto high = highPass.generateEvents();
                //lowPass.accept(high);
                //const auto low = lowPass.generateEvents();
                //maskFilter.accept(*raw);
                //const auto masked = maskFilter.generateEvents();
                updateImageAndTimestamps(
                    accumulatorTimeConstant,
                    accumulatorGain,
                    resolution.height,
                    resolution.width,
                    interpolationMap,
                    high,
                    image,
                    timestamps);
                //eventBuffer.add(masked);
                //accumulator.accept(masked);
                for(dv::Event ev : high) {
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