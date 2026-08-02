cmake_minimum_required(VERSION 3.10)
project(SlamDemo)

find_package(dv-processing REQUIRED)
find_package(OpenCV REQUIRED)

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

add_executable(SlamDemo src/slamDemo.cpp)
target_include_directories(SlamDemo PRIVATE src)
add_executable(CopyPaste src/inivation_copypaste.cpp)

target_link_libraries(SlamDemo PUBLIC
    dv::processing
    opencv_core
    cnpy_imported
)
target_link_libraries(CopyPaste PUBLIC
    dv::processing
    opencv_core
    cnpy_imported
)