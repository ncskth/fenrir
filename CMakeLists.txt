set(CMAKE_CXX_COMPILER "/usr/bin/g++-13")
cmake_minimum_required(VERSION 3.10)
project(Fenrir)

set(OpenCV_FIND_REQUIRED_COMPONENTS core imgproc highgui)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(dv-processing REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui viz calib3d)
find_package(cppzmq REQUIRED)
find_package(fmt REQUIRED)

# Find cnpy manually
find_path(cnpy_INCLUDE_DIR NAMES cnpy.h PATHS /usr/local/include)
find_library(cnpy_LIBRARY NAMES cnpy PATHS /usr/local/lib)

if(cnpy_INCLUDE_DIR AND cnpy_LIBRARY)
    add_library(cnpy_imported UNKNOWN IMPORTED)
    set_target_properties(cnpy_imported PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${cnpy_INCLUDE_DIR}
        IMPORTED_LOCATION ${cnpy_LIBRARY}
    )
else()
    message(FATAL_ERROR "cnpy not found")
endif()

add_executable(DepthDemo src/depthDemo.cpp
    src/visualization.cpp
    src/cameraCapture.cpp
    src/disparityEstimation.cpp)
target_include_directories(DepthDemo PRIVATE include)

add_executable(VisClient src/visualizationClient.cpp
    src/visualization.cpp)
target_include_directories(VisClient PRIVATE include)

add_executable(IMUBias src/printImuBias.cpp)

target_link_libraries(DepthDemo PUBLIC
    dv::processing
    ${OpenCV_LIBS}    # Changed from opencv_core
    cnpy_imported
    Boost::program_options
    cppzmq
    fmt::fmt
)
target_link_libraries(VisClient PUBLIC
    dv::processing
    ${OpenCV_LIBS}    # Changed from opencv_core
    cnpy_imported
    Boost::program_options
    cppzmq
    fmt::fmt
)

target_link_libraries(IMUBias PUBLIC
    dv::processing
)