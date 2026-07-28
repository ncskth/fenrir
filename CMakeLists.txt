PROJECT(SlamDemo)
ADD_SUBDIRECTORY(src)
ADD_EXECUTABLE(SlamDemo src/slam_demo.cpp)

# Find installed dv-processing.
FIND_PACKAGE(dv-processing REQUIRED)

# Link your targets against the library
TARGET_LINK_LIBRARIES(SlamDemo PUBLIC
                      dv::processing
                      opencv_core)