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

#include <opencv2/highgui.hpp>

int main() {
    using namespace std::chrono_literals;

    // Open any camera
    auto capture = dv::io::camera::open();

    // Make sure it supports event stream output, throw an error otherwise
    if (!capture->isEventStreamAvailable()) {
        throw dv::exceptions::RuntimeError("Input camera does not provide an event stream.");
    }

    // Initialize an accumulator with some resolution
    dv::Accumulator accumulator(*capture->getEventResolution());

    // Apply configuration, these values can be modified to taste
    accumulator.setMinPotential(0.f);
    accumulator.setMaxPotential(1.f);
    accumulator.setNeutralPotential(0.5f);
    accumulator.setEventContribution(0.15f);
    accumulator.setDecayFunction(dv::Accumulator::Decay::EXPONENTIAL);
    accumulator.setDecayParam(1e+6);
    accumulator.setIgnorePolarity(false);
    accumulator.setSynchronousDecay(false);

    // Initialize a preview window
    cv::namedWindow("Preview", cv::WINDOW_NORMAL);

    // Initialize a slicer
    dv::EventStreamSlicer slicer;

    // Register a callback every 33 milliseconds
    slicer.doEveryTimeInterval(33ms, [&accumulator](const dv::EventStore &events) {
        // Pass events into the accumulator and generate a preview frame
        accumulator.accept(events);
        dv::Frame frame = accumulator.generateFrame();

        // Show the accumulated image
        cv::imshow("Preview", frame.image);
        cv::waitKey(2);
    });

    // Run the event processing while the camera is connected
    while (capture->isRunning()) {
        // Receive events, check if anything was received
        if (const auto events = capture->getNextEventBatch()) {
            // If so, pass the events into the slicer to handle them
            slicer.accept(*events);
        }
    }

    return 0;
}