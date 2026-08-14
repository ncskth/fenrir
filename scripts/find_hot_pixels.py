import os

from os.path import join
from datetime import timedelta
from time import time

import argparse
import cv2 as cv
import numpy as np
import dv_processing as dv


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--calib_json", type=str, required=True)
    parser.add_argument("--output_directory", type=str, default=os.getcwd())
    parser.add_argument("--hot_threshold", type=int, default=100)
    args = parser.parse_args()

    calibration = dv.camera.CalibrationSet.LoadFromFile(args.calib_json)

    # Open cameras
    capture_left = dv.io.camera.open(calibration.getCameraCalibrations()["C0"].name)
    capture_right = dv.io.camera.open(calibration.getCameraCalibrations()["C1"].name)

    # Make sure it supports event stream output, throw an error otherwise
    if not (capture_left.isEventStreamAvailable() and capture_right.isEventStreamAvailable()):
        raise RuntimeError("Input camera does not provide an event stream.")

    # Initialize an accumulator with some resolution
    visualizer_left = dv.visualization.EventVisualizer(capture_left.getEventResolution())

    # Apply color scheme configuration, these values can be modified to taste
    visualizer_left.setBackgroundColor(dv.visualization.colors.black())
    visualizer_left.setPositiveColor(dv.visualization.colors.blue())
    visualizer_left.setNegativeColor(dv.visualization.colors.green())

    # Initialize an accumulator with some resolution
    visualizer_right = dv.visualization.EventVisualizer(capture_right.getEventResolution())

    # Apply color scheme configuration, these values can be modified to taste
    visualizer_right.setBackgroundColor(dv.visualization.colors.black())
    visualizer_right.setPositiveColor(dv.visualization.colors.iniBlue())
    visualizer_right.setNegativeColor(dv.visualization.colors.green())

    # Initialize a preview window
    cv.namedWindow("Left", cv.WINDOW_NORMAL)
    cv.namedWindow("Right", cv.WINDOW_NORMAL)

    # Initialize a slicer
    slicer_left = dv.EventStreamSlicer()
    slicer_right = dv.EventStreamSlicer()
    high_pass_left = dv.noise.BackgroundActivityNoiseFilter((640, 480), backgroundActivityDuration=timedelta(milliseconds=0.01))
    high_pass_right = dv.noise.BackgroundActivityNoiseFilter((640, 480), backgroundActivityDuration=timedelta(milliseconds=0.01))
    low_pass_left = dv.noise.LowPassFilter((640, 480), 500)
    low_pass_right = dv.noise.LowPassFilter((640, 480), 500)

    # Declare the callback method for slicer
    def slicing_callback_left(events: dv.EventStore):
        # Generate a preview frame
        frame = visualizer_left.generateImage(events)

        # Show the accumulated image
        cv.imshow("Left", frame)
        cv.waitKey(2)
    def slicing_callback_right(events: dv.EventStore):
        # Generate a preview frame
        frame = visualizer_right.generateImage(events)

        # Show the accumulated image
        cv.imshow("Right", frame)
        cv.waitKey(2)


    # Register callback to be performed every 33 milliseconds
    slicer_left.doEveryTimeInterval(timedelta(milliseconds=30), slicing_callback_left)
    slicer_right.doEveryTimeInterval(timedelta(milliseconds=30), slicing_callback_right)

    def eventstore_to_numpy(events: dv.EventStore):
        return np.stack(([e.timestamp() for e in events],
                        [e.x() for e in events],
                        [e.y() for e in events],
                        [e.polarity() for e in events]), axis=-1)

    accumulator_left = np.zeros((640, 480))
    accumulator_right = np.zeros((640, 480))

    start = time()

    # Run the event processing while the camera is connected
    while capture_left.isRunning() and capture_right.isRunning() and time() - start < 10:
        # Receive events
        events_left = capture_left.getNextEventBatch()
        events_right = capture_right.getNextEventBatch()

        # Check if anything was received
        if events_left is not None:
            slicer_left.accept(events_left)
            for event in events_left:
                accumulator_left[event.x(), event.y()] += 1
        if events_right is not None:
            slicer_right.accept(events_right)
            for event in events_right:
                accumulator_right[event.x(), event.y()] += 1

    hot_pixels_left_x, hot_pixels_left_y = np.where(accumulator_left > args.hot_threshold)
    np.save(join(args.output_directory, "hot_pixels_left_x.npy"), hot_pixels_left_x)
    np.save(join(args.output_directory, "hot_pixels_left_y.npy"), hot_pixels_left_y)
    print(f"Left found {len(hot_pixels_left_x)} hot pixels")
    hot_pixels_right_x, hot_pixels_right_y = np.where(accumulator_right > args.hot_threshold)
    np.save(join(args.output_directory, "hot_pixels_right_x.npy"), hot_pixels_right_x)
    np.save(join(args.output_directory, "hot_pixels_right_y.npy"), hot_pixels_right_y)
    print(f"Right found {len(hot_pixels_right_x)} hot pixels")