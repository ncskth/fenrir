#include <dv-processing/io/camera/discovery.hpp>
#include <dv-processing/visualization/event_visualizer.hpp>
#include <dv-processing/data/generate.hpp>
#include <dv-processing/noise/background_activity_noise_filter.hpp>
#include <dv-processing/noise/frequency_filters.hpp>
#include <dv-processing/core/core.hpp>
#include <dv-processing/camera/calibration_set.hpp>
#include <dv-processing/core/stereo_event_stream_slicer.hpp>
#include <dv-processing/depth/semi_dense_stereo_matcher.hpp>

#include <opencv2/highgui.hpp>

#include <iostream>
#include <queue>

using namespace std;
using namespace std::chrono_literals;

cv::Mat adaptiveAccumulation(cv::Size resolution,
                             int x_patches,
                             int y_patches,
                             //cv::Array undistort_map1,
                             //cv::Array undistort_map2,
                             const dv::EventStore events) {

    int n_events = events.size();

    cv::Mat representation_AA = cv::Mat::zeros(resolution, CV_8U);   //for temporal stereo matching
    cv::Mat AA_frequency = cv::Mat::zeros(resolution, CV_8U);   //for point sampling

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
        //cout << "e.y " << ey << endl;
        //cout << "e.x " << ex << endl;
        int y = (int)ey / (int)ceil((double)resolution.height / (double)y_patches);
        int x = (int)ex / (int)ceil((double)resolution.width / (double)x_patches);
        ////cout << "y " << y << endl;
        ////cout << "x " << x << endl;
        beta[y * x_patches + x] = 1 / (1 + final_activity[y * x_patches + x] * abs(ts*1e-6 - last_event_time[y * x_patches + x])); // eq. 2
        ////cout << "eq 2" << endl;
        if (y * x_patches + x >= x_patches * y_patches)
          exit(-1);
        final_activity[y * x_patches + x] = beta[y * x_patches + x] * final_activity[y * x_patches + x] + 1; // eq. 1
        ////cout << "eq 1" << endl;
        last_event_time[y * x_patches + x] = ts*1e-6;
        ////cout << "updated time`" << endl;
        // nums_temp[y * x_patches + x]++;
    }
    //cout << "finished first loop" << endl;
    // for(int i = 0; i < x_patches * y_patches; i++)
    // final_activity[i] = std::sqrt(1 / (0.01 / nums_temp[i]));  // eq. 3

    fill(beta.begin(), beta.end(), 0);
    fill(last_event_time.begin(), last_event_time.end(), 0);
    for (int i = n_events - 1; i >= 0; i--) // traverse events in reverse to accumulate the latest events
    {
        int64_t ts = events[i].timestamp();
        int16_t ex = events[i].x();
        int16_t ey = events[i].y();
        //cout << "e.y " << ey << endl;
        //cout << "e.x " << ex << endl;
        int y = (int)ey / (int)ceil((double)resolution.height / (double)y_patches);
        int x = (int)ex / (int)ceil((double)resolution.width / (double)x_patches);
        ////cout << "y " << y << endl;
        ////cout << "x " << x << endl;
        if (flag[y * x_patches + x] != true)
            continue;
        beta[y * x_patches + x] = 1 / (1 + event_activity[y * x_patches + x] * abs(ts*1e-6 - last_event_time[y * x_patches + x])); // eq. 2
        ////cout << "eq 2" << endl;
        event_activity[y * x_patches + x] = beta[y * x_patches + x] * event_activity[y * x_patches + x] + 1;                            // eq. 1
        ////cout << "eq 1" << endl;
        last_event_time[y * x_patches + x] = ts*1e-6;
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
    //cout << "finished second loop" << endl;

    //distortion correction
    //cv::remap(representation_AA, representation_AA, undistort_map1, undistort_map2, CV_INTER_LINEAR);
    return representation_AA;
}

template<typename T>
class SafeQueue {
    queue<T> q;
    mutex m;
public:
    void push(const T& val) {
        lock_guard<mutex> lock(m);
        q.push(val);
    }
    T pop() {
        lock_guard<mutex> lock(m);
        auto val = q.front();
        q.pop();
        return val;
    }
    bool empty() {
        lock_guard<mutex> lock(m);
        return q.empty();
    }
};

vector<cv::Mat> leftImageRepresentation(const cv::Size resolution,
                                        auto& positive,
                                        auto& negative,
                                        dv::Accumulator& positiveTS,
                                        dv::Accumulator& negativeTS,
                                        dv::EventStore& events) {
        auto aaFuture = async(launch::async, [&]() {
            return adaptiveAccumulation(resolution, 8, 6, events);
        });
        auto negFuture = async(launch::async, [&]() {
            negative.accept(events);
            const auto negativeEvents = negative.generateEvents();
            negativeTS.accept(negativeEvents);
            cv::Mat image = cv::Scalar(255) - negativeTS.generateFrame().image;
            return image;
        });

        // Update positive time surface on main thread
        positive.accept(events);
        const auto positiveEvents = positive.generateEvents();
        positiveTS.accept(positiveEvents);
        cv::Mat posTS = positiveTS.generateFrame().image;

        cv::Mat aa = aaFuture.get();
        cv::Mat negTS = negFuture.get();

        // Combine and display
        vector<cv::Mat> images(3);
        images[0] = posTS;
        images[1] = negTS;
        images[2] = aa;
        return images;
        //cv::Mat leftImage;
        //cv::hconcat(images, leftImage);
        //cv::imshow("Left", leftImage);
        //cv::waitKey(2);
}

vector<cv::Mat> rightImageRepresentation(auto& positive,
                                         auto& negative,
                                         dv::Accumulator& positiveTS,
                                         dv::Accumulator& negativeTS,
                                         dv::EventStore& events) {
        auto negFuture = async(launch::async, [&]() {
            negative.accept(events);
            const auto negativeEvents = negative.generateEvents();
            negativeTS.accept(negativeEvents);
            cv::Mat image = cv::Scalar(255) - negativeTS.generateFrame().image;
            return image;
        });

        // Update positive time surface on main thread
        positive.accept(events);
        const auto positiveEvents = positive.generateEvents();
        positiveTS.accept(positiveEvents);
        cv::Mat posTS = positiveTS.generateFrame().image;

        cv::Mat negTS = negFuture.get();

        // Combine and display
        vector<cv::Mat> images(3);
        images[0] = posTS;
        images[1] = negTS;
        //cv::Mat rightImage;
        //cv::hconcat(images, rightImage);
        return images;
        //cv::imshow("Right", rightImage);
        //cv::waitKey(2);
}

void imageRepresentationCallback(bool isLeft,
                                 const cv::Size resolution,
                                 dv::EventPolarityFilter<>& positive,
                                 dv::EventPolarityFilter<>& negative,
                                 dv::Accumulator& positiveTS,
                                 dv::Accumulator& negativeTS,
                                 queue<dv::EventStore>& incomingEvents,
                                 queue<vector<cv::Mat>>& outgoingImages
                                 ) {

    while(true) {
        if(!incomingEvents.empty()) {
            auto events = incomingEvents.front();
            incomingEvents.pop();

            if(isLeft) {
                vector<cv::Mat> images = leftImageRepresentation(resolution, positive, negative, positiveTS, negativeTS, events);
                outgoingImages.push(images);
            } else {
                vector<cv::Mat> images = rightImageRepresentation(positive, negative, positiveTS, negativeTS, events);
                outgoingImages.push(images);
            }
        } else {
            this_thread::sleep_for(2ms);
        }
    }
}

class ImageRepresentation {
private:
    bool is_left;
    string hot_pixels_dir;
    cv::Size resolution;
    std::queue<dv::EventStore>& events_in;
    std::queue<dv::EventStore>& events_out;
    std::thread processing_thread;
    dv::noise::BackgroundActivityNoiseFilter<dv::EventStore> high_pass;
    dv::noise::LowPassFilter<dv::EventStore> low_pass;

    void eventProcessingCallback() {
        while(true) {
            if(!events_in.empty()) {
                auto in_events = events_in.front();
                events_in.pop();
                high_pass.accept(in_events);
                low_pass.accept(high_pass.generateEvents());
                events_out.push(low_pass.generateEvents());
            }
        }
    }

public:
    ImageRepresentation(bool left, string hot_pixels, cv::Size r,
                       std::queue<dv::EventStore>& inq,
                       std::queue<dv::EventStore>& outq)
        : is_left(left), hot_pixels_dir(hot_pixels), resolution(r),
          events_in(inq), events_out(outq),
          high_pass(resolution, 10us), low_pass(resolution, 500.0f),
          processing_thread(&ImageRepresentation::eventProcessingCallback, this) {}
};