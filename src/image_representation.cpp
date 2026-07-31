#include <dv-processing/data/generate.hpp>
#include <dv-processing/noise/background_activity_noise_filter.hpp>
#include <dv-processing/noise/frequency_filters.hpp>
#include <dv-processing/core/core.hpp>

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
                             const dv::EigenEvents events) {

    int n_events = events.timestamps.rows();

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
        int64_t ts = events.timestamps(i, 0);
        int16_t ex = events.coordinates(i, 0);
        int16_t ey = events.coordinates(i, 1);
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
        int64_t ts = events.timestamps(i, 0);
        int16_t ex = events.coordinates(i, 0);
        int16_t ey = events.coordinates(i, 1);
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

void imageRepresentationCallback(bool is_left,
                                 cv::Size resolution,
                                 SafeQueue<dv::EventStore>& events_in,
                                 SafeQueue<dv::EventStore>& events_out,
                                 condition_variable& cv,
                                 mutex& mtx,
                                 bool& ready_flag
                                 ) {

    dv::noise::BackgroundActivityNoiseFilter high_pass(resolution, 10us);
    dv::noise::LowPassFilter low_pass(resolution, 500.0f);

    while(true) {
        if(!events_in.empty()) {
            auto in_events = events_in.pop();

            ////cout << (is_left ? "true" : "false") << " events_in pop\n";
            high_pass.accept(in_events);
            low_pass.accept(high_pass.generateEvents());

            events_out.push(low_pass.generateEvents());

            // Notify main thread this side is ready
            {
                lock_guard<mutex> lock(mtx);
                ready_flag = true;
            }
            cv.notify_one();

            ////cout << (is_left ? "true" : "false") << " events_out push\n";
        } else {
            this_thread::yield();
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