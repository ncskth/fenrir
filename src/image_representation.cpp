#include <dv-processing/data/generate.hpp>
#include <dv-processing/noise/background_activity_noise_filter.hpp>
#include <dv-processing/noise/frequency_filters.hpp>
#include <dv-processing/core/core.hpp>

#include <opencv2/highgui.hpp>

#include <iostream>
#include <queue>

using namespace std;

using namespace std::chrono_literals;

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

            //cout << (is_left ? "true" : "false") << " events_in pop\n";
            high_pass.accept(in_events);
            low_pass.accept(high_pass.generateEvents());

            events_out.push(low_pass.generateEvents());

            // Notify main thread this side is ready
            {
                lock_guard<mutex> lock(mtx);
                ready_flag = true;
            }
            cv.notify_one();

            //cout << (is_left ? "true" : "false") << " events_out push\n";
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