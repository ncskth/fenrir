from datetime import timedelta

import cv2 as cv
import numpy as np
import dv_processing as dv

# Open cameras
capture_left = dv.io.camera.open("DXM00089")
capture_right = dv.io.camera.open("DXM00090")

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

# background noise filter
high_pass_left = dv.noise.BackgroundActivityNoiseFilter((640, 480), backgroundActivityDuration=timedelta(milliseconds=0.01))
high_pass_right = dv.noise.BackgroundActivityNoiseFilter((640, 480), backgroundActivityDuration=timedelta(milliseconds=0.01))

# refractory period filter
low_pass_left = dv.noise.LowPassFilter((640, 480), 500)
low_pass_right = dv.noise.LowPassFilter((640, 480), 500)

# hot pixel filter
mask_left = 255*np.ones((640, 480), dtype=np.uint8)
mask_right = 255*np.ones((640, 480), dtype=np.uint8)
hot_pixels_left_x = np.load("hot_pixels_left_x.npy")
hot_pixels_left_y = np.load("hot_pixels_left_y.npy")
hot_pixels_right_x = np.load("hot_pixels_right_x.npy")
hot_pixels_right_y = np.load("hot_pixels_right_y.npy")
mask_left[hot_pixels_left_x, hot_pixels_left_y] = 0
mask_right[hot_pixels_right_x, hot_pixels_right_y] = 0
print(np.sum((255 - mask_left)/255))
print(np.sum((255 - mask_right)/255))
cold_pass_left = dv.EventMaskFilter(mask_left.T)
cold_pass_right = dv.EventMaskFilter(mask_right.T)

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
#filter_left.doEveryTimeInterval(timedelta(milliseconds=10), slicing_callback_left)
#filter_right.doEveryTimeInterval(timedelta(milliseconds=10), slicing_callback_right)

# Run the event processing while the camera is connected
while capture_left.isRunning() and capture_right.isRunning():
    # Receive events
    events_left = capture_left.getNextEventBatch()
    events_right = capture_right.getNextEventBatch()

    # Check if anything was received
    if events_left is not None:
        # If so, pass the events into the slicer to handle them
        high_pass_left.accept(events_left)
        events_left = high_pass_left.generateEvents()
        low_pass_left.accept(events_left)
        events_left = low_pass_left.generateEvents()
        cold_pass_left.accept(events_left)
        events_left = cold_pass_left.generateEvents()
        slicer_left.accept(events_left)
    if events_right is not None:
        high_pass_right.accept(events_right)
        events_right = high_pass_right.generateEvents()
        low_pass_right.accept(events_right)
        events_right = low_pass_right.generateEvents()
        cold_pass_right.accept(events_right)
        events_right = cold_pass_right.generateEvents()
        slicer_right.accept(events_right)
