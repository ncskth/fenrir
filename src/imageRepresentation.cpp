#include <imageRepresentation.h>

namespace SlamDemo {

    using namespace std;
    using namespace std::chrono_literals;

    cv::Mat adaptiveAccumulation(
        cv::Size resolution,
        int x_patches,
        int y_patches,
        const dv::EventStore events)
    {

        int n_events = events.size();

        cv::Mat representation_AA = cv::Mat::zeros(resolution, CV_8U); // for temporal stereo matching
        cv::Mat AA_frequency = cv::Mat::zeros(resolution, CV_8U);      // for point sampling

        std::vector<double> last_activity(x_patches * y_patches, 0), event_activity(x_patches * y_patches, 0), beta(x_patches * y_patches, 0);
        std::vector<double> last_event_time(x_patches * y_patches, 0);
        std::vector<bool> flag(x_patches * y_patches, true);
        int flags = 0;
        double conv_thresh_ = 0.95; // convergence threshold
        std::vector<double> final_activity(x_patches * y_patches, 0);
        std::vector<int> num(x_patches * y_patches, 0);

        // std::vector<int> nums_temp(x_patches * y_patches, 0);
        int nums_EQ = 0;
        // calculate the final activity by all events, also can be estimated by eq. 3 in the paper
        for (size_t i = 0; i < n_events; i++)
        {
            int64_t ts = events[i].timestamp();
            int16_t ex = events[i].x();
            int16_t ey = events[i].y();
            ////cout << "height " << resolution.height << endl;
            ////cout << "width " << resolution.width << endl;
            // cout << "e.y " << ey << endl;
            // cout << "e.x " << ex << endl;
            int y = (int)ey / (int)ceil((double)resolution.height / (double)y_patches);
            int x = (int)ex / (int)ceil((double)resolution.width / (double)x_patches);
            ////cout << "y " << y << endl;
            ////cout << "x " << x << endl;
            beta[y * x_patches + x] = 1 / (1 + final_activity[y * x_patches + x] * abs(ts * 1e-6 - last_event_time[y * x_patches + x])); // eq. 2
            ////cout << "eq 2" << endl;
            if (y * x_patches + x >= x_patches * y_patches)
                exit(-1);
            final_activity[y * x_patches + x] = beta[y * x_patches + x] * final_activity[y * x_patches + x] + 1; // eq. 1
            ////cout << "eq 1" << endl;
            last_event_time[y * x_patches + x] = ts * 1e-6;
            ////cout << "updated time`" << endl;
            // nums_temp[y * x_patches + x]++;
        }
        // cout << "finished first loop" << endl;
        //  for(int i = 0; i < x_patches * y_patches; i++)
        //  final_activity[i] = std::sqrt(1 / (0.01 / nums_temp[i]));  // eq. 3

        fill(beta.begin(), beta.end(), 0);
        fill(last_event_time.begin(), last_event_time.end(), 0);
        for (int i = n_events - 1; i >= 0; i--) // traverse events in reverse to accumulate the latest events
        {
            int64_t ts = events[i].timestamp();
            int16_t ex = events[i].x();
            int16_t ey = events[i].y();
            // cout << "e.y " << ey << endl;
            // cout << "e.x " << ex << endl;
            int y = (int)ey / (int)ceil((double)resolution.height / (double)y_patches);
            int x = (int)ex / (int)ceil((double)resolution.width / (double)x_patches);
            ////cout << "y " << y << endl;
            ////cout << "x " << x << endl;
            if (flag[y * x_patches + x] != true)
                continue;
            beta[y * x_patches + x] = 1 / (1 + event_activity[y * x_patches + x] * abs(ts * 1e-6 - last_event_time[y * x_patches + x])); // eq. 2
            ////cout << "eq 2" << endl;
            event_activity[y * x_patches + x] = beta[y * x_patches + x] * event_activity[y * x_patches + x] + 1; // eq. 1
            ////cout << "eq 1" << endl;
            last_event_time[y * x_patches + x] = ts * 1e-6;
            ////cout << "updated last event time" << endl;
            AA_frequency.at<uchar>(ey, ex)++;
            num[y * x_patches + x]++;
            if (AA_frequency.at<uchar>(ey, ex) >= 1)
                representation_AA.at<uchar>(ey, ex) = 255;
            if (num[y * x_patches + x] >= 10) // each patch is checked for convergence once every ten events accumulated
            {
                if (last_activity[y * x_patches + x] != 0)
                {
                    if ((abs(event_activity[y * x_patches + x] - final_activity[y * x_patches + x])) < conv_thresh_)
                    {
                        flag[y * x_patches + x] = false;
                        flags++;
                        if (flags == x_patches * y_patches)
                            break;
                        else
                            continue;
                    }
                }
                last_activity[y * x_patches + x] = event_activity[y * x_patches + x];
                num[y * x_patches + x] = 0;
            }
        }

        return representation_AA;
    }

    cv::Mat leftImageRepresentation(
        const cv::Size resolution,
        const int aaPatchesX,
        const int aaPatchesY,
        const int downsampleFactor,
        const cv::Matx33f cameraMatrix,
        const vector<float> distortionCoefficients,
        const int tsDecayMs,
        vector<int64_t>& lastPosts,
        vector<int64_t>& lastNegts,
        dv::EventStore &events)
    {
        int64_t recentTime = events[events.size() - 1].timestamp();
        for(auto ev : events) {
            if(ev.polarity()) {
                lastPosts[ev.x() + resolution.width*ev.y()] = ev.timestamp();
            }
            else {
                lastNegts[ev.x() + resolution.width*ev.y()] = ev.timestamp();
            }
        }
        cv::Mat posTSDistorted = cv::Mat(resolution, CV_64FC1);
        cv::Mat negTSDistorted = cv::Mat(resolution, CV_64FC1);
        for(size_t i = 0; i < resolution.area(); i++) {
            posTSDistorted.at<double>(i/resolution.width, i%resolution.width) = exp((1e-3*(lastPosts[i] - recentTime)/tsDecayMs));
            negTSDistorted.at<double>(i/resolution.width, i%resolution.width) = exp((1e-3*(lastNegts[i] - recentTime)/tsDecayMs));
        }
        cv::Mat combinedTSDistorted = posTSDistorted - negTSDistorted;
        cv::Mat combinedTS;
        cv::undistort(combinedTSDistorted, combinedTS, cameraMatrix, distortionCoefficients);

        return combinedTS;
        /*
        cv::Mat image = adaptiveAccumulation(resolution, aaPatchesX, aaPatchesY, events);
        cv::Mat aa;
        cv::undistort(image, aa, cameraMatrix, distortionCoefficients);

        vector<dv::Event> dx_events, dy_events;
        for (size_t i = 0; i < events.size(); i += downsampleFactor)
        {
            dv::Event ev = events[i];
            if (aa.at<uchar>(ev.y(), ev.x()) > 200 && ev.x() > 0 && ev.x() < resolution.width - 1 && ev.y() > 0 && ev.y() < resolution.height - 1)
            {
                auto sobel = sobelAtPoint(aa, ev.x(), ev.y());
                if (abs(get<0>(sobel)) > abs(get<1>(sobel)))
                {
                    dx_events.push_back(ev);
                }
                else if (abs(get<1>(sobel)) > abs(get<0>(sobel)))
                {
                    dy_events.push_back(ev);
                }
            }
        }

        vector<cv::Mat> result = tsFuture.get();
        result.push_back(aa);

        return {result, dx_events, dy_events};
        */
    }

    cv::Mat imageRepresentation(
        const cv::Size resolution,
        const cv::Matx33f cameraMatrix,
        const vector<float> distortionCoefficients,
        const int tsDecayMs,
        vector<int64_t>& lastPosts,
        vector<int64_t>& lastNegts,
        dv::EventStore &events)
    {
        int64_t recentTime = events[events.size() - 1].timestamp();
        for(auto ev : events) {
            if(ev.polarity()) {
                lastPosts[ev.x() + resolution.width*ev.y()] = ev.timestamp();
            }
            else {
                lastNegts[ev.x() + resolution.width*ev.y()] = ev.timestamp();
            }
        }
        cv::Mat posTSDistorted = cv::Mat(resolution, CV_64FC1);
        cv::Mat negTSDistorted = cv::Mat(resolution, CV_64FC1);
        for(size_t i = 0; i < resolution.area(); i++) {
            posTSDistorted.at<double>(i/resolution.width, i%resolution.width) = exp((1e-3*(lastPosts[i] - recentTime)/tsDecayMs));
            negTSDistorted.at<double>(i/resolution.width, i%resolution.width) = exp((1e-3*(lastNegts[i] - recentTime)/tsDecayMs));
        }
        cv::Mat combinedTSDistorted = posTSDistorted - negTSDistorted;
        cv::Mat combinedTS;
        cv::undistort(combinedTSDistorted, combinedTS, cameraMatrix, distortionCoefficients);

        return combinedTS;
    }

    void leftImageRepresentationLoop(
        const cv::Size resolution,
        const int aaPatchesX,
        const int aaPatchesY,
        const int downsampleFactor,
        const int timeSurfaceMilliseconds,
        const cv::Matx33f &cameraMatrix,
        const vector<float> &distortionCoefficients,
        queue<dv::EventStore> &incomingEvents,
        queue<cv::Mat> &outgoingImages1,
        queue<cv::Mat> &outgoingImages2)
    {

        vector<int64_t> lastPositiveTimestamps(resolution.area());
        vector<int64_t> lastNegativeTimestamps(resolution.area());

        while (true)
        {
            if (!incomingEvents.empty())
            {
                auto events = incomingEvents.front();
                incomingEvents.pop();

                auto image = leftImageRepresentation(
                    resolution,
                    aaPatchesX,
                    aaPatchesY,
                    downsampleFactor,
                    cameraMatrix,
                    distortionCoefficients,
                    timeSurfaceMilliseconds,
                    ref(lastPositiveTimestamps),
                    ref(lastNegativeTimestamps),
                    events
                );
                cv::Mat imageScaled = 127.*image + 128.;
                cv::Mat imageToRender;
                imageScaled.convertTo(imageToRender, CV_8UC1);
                outgoingImages1.push(imageToRender);
                //outgoingImages2.push(image);
            }
            else
            {
                this_thread::sleep_for(2ms);
            }
        }
    }

    void imageRepresentationLoop(
        const cv::Size resolution,
        const int timeSurfaceMilliseconds,
        const cv::Matx33f &cameraMatrix,
        const vector<float> &distortionCoefficients,
        queue<dv::EventStore> &incomingEvents,
        queue<cv::Mat> &outgoingImages1,
        queue<cv::Mat> &outgoingImages2)
    {

        vector<int64_t> lastPositiveTimestamps(resolution.area());
        vector<int64_t> lastNegativeTimestamps(resolution.area());

        while (true)
        {
            if (!incomingEvents.empty())
            {
                auto events = incomingEvents.front();
                incomingEvents.pop();

                cv::Mat image = imageRepresentation(
                    resolution,
                    cameraMatrix,
                    distortionCoefficients,
                    timeSurfaceMilliseconds,
                    ref(lastPositiveTimestamps),
                    ref(lastNegativeTimestamps),
                    events
                );
                cv::Mat imageScaled = 127.*image + 128.;
                cv::Mat imageToRender;
                imageScaled.convertTo(imageToRender, CV_8UC1);
                outgoingImages1.push(imageToRender);
                outgoingImages2.push(image);
            }
            else
            {
                this_thread::sleep_for(2ms);
            }
        }
    }
}