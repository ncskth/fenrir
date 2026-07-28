#include <dv-processing/io/camera/discovery.hpp>
#include <dv-processing/visualization/event_visualizer.hpp>
#include <dv-processing/data/generate.hpp>
#include <dv-processing/noise/background_activity_noise_filter.hpp>
#include <dv-processing/noise/frequency_filters.hpp>
#include <future>

#include <opencv2/highgui.hpp>

int main()
{
    using namespace std::chrono_literals;
    using namespace std;

    // Open any camera
    auto capture_left = dv::io::camera::open("DXM00089");
    auto capture_right = dv::io::camera::open("DXM00090");

    // Make sure it supports event stream output, throw an error otherwise
    if (!capture_left->isEventStreamAvailable() || !capture_right->isEventStreamAvailable())
    {
        throw dv::exceptions::RuntimeError("Input camera does not provide an event stream.");
    }

    // Initialize an accumulator with some resolution
    auto resolution = *capture_left->getEventResolution();
    dv::visualization::EventVisualizer visualizer_left(resolution);
    dv::visualization::EventVisualizer visualizer_right(resolution);

    // Apply color scheme configuration, these values can be modified to taste
    visualizer_left.setBackgroundColor(dv::visualization::colors::black);
    visualizer_left.setPositiveColor(dv::visualization::colors::blue);
    visualizer_left.setNegativeColor(dv::visualization::colors::green);
    visualizer_right.setBackgroundColor(dv::visualization::colors::black);
    visualizer_right.setPositiveColor(dv::visualization::colors::blue);
    visualizer_right.setNegativeColor(dv::visualization::colors::green);

    // Initialize a preview window
    cv::namedWindow("Left", cv::WINDOW_NORMAL);
    cv::namedWindow("Right", cv::WINDOW_NORMAL);

    // Initialize a slicer
    dv::EventStreamSlicer slicer_left;
    dv::EventStreamSlicer slicer_right;

    // Register a callback every 33 milliseconds
    slicer_left.doEveryTimeInterval(50ms, [&visualizer_left](const dv::EventStore &events)
                                    {
        // Generate a preview frame
        cv::Mat image = visualizer_left.generateImage(events);

        // Show the accumulated image
        cv::imshow("Left", image);
        cv::waitKey(2); });
    slicer_right.doEveryTimeInterval(50ms, [&visualizer_right](const dv::EventStore &events)
                                     {
        // Generate a preview frame
        cv::Mat image = visualizer_right.generateImage(events);

        // Show the accumulated image
        cv::imshow("Right", image);
        cv::waitKey(2); });

    // Initialize a background activity noise filter with 0.01-millisecond activity period
    dv::noise::BackgroundActivityNoiseFilter high_pass_left(resolution, 1ms);
    dv::noise::BackgroundActivityNoiseFilter high_pass_right(resolution, 1ms);
    dv::noise::LowPassFilter low_pass_left(resolution, 500.0f);
    dv::noise::LowPassFilter low_pass_right(resolution, 500.0f);

    // Run the event processing while the camera is connected
    /*
    // Sequential version which I thought I could speed up
    while (capture_left->isRunning() && capture_right->isRunning())
    {

        if (const auto events_left = capture_left->getNextEventBatch())
        {
                high_pass_left.accept(*events_left);
                const dv::EventStore filtered1 = high_pass_left.generateEvents();
                low_pass_left.accept(filtered1);
                const dv::EventStore filtered2 = low_pass_left.generateEvents();
                slicer_left.accept(filtered2);
        }

        //*
        if (const auto events_right = capture_right->getNextEventBatch())
        {
            high_pass_right.accept(*events_right);
            const dv::EventStore filtered1 = high_pass_right.generateEvents();
            low_pass_right.accept(filtered1);
            const dv::EventStore filtered2 = low_pass_right.generateEvents();
            slicer_right.accept(filtered2);
        }
    }

    //*/
    /*
    while (capture_left->isRunning() && capture_right->isRunning()) {
        const auto events_left = capture_left->getNextEventBatch();
        const auto events_right = capture_right->getNextEventBatch();

        auto left_future = std::async(std::launch::async, [&]() -> std::optional<dv::EventStore> {
            if (events_left) {
                high_pass_left.accept(*events_left);
                const dv::EventStore filtered1 = high_pass_left.generateEvents();
                low_pass_left.accept(filtered1);
                const dv::EventStore filtered2 = low_pass_left.generateEvents();
                return filtered2;
            }
            return std::nullopt;
        });

        auto right_future = std::async(std::launch::async, [&]() -> std::optional<dv::EventStore> {
            if (events_right) {
                high_pass_right.accept(*events_right);
                const dv::EventStore filtered1 = high_pass_right.generateEvents();
                low_pass_right.accept(filtered1);
                const dv::EventStore filtered2 = low_pass_right.generateEvents();
                return filtered2;
            }
            return std::nullopt;
        });

        if (auto filtered = left_future.get()) {
            slicer_left.accept(*filtered);
        }
        if (auto filtered = right_future.get()) {
            slicer_right.accept(*filtered);
        }
    }
    return 0;
    */
}