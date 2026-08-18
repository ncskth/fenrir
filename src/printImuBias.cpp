#include <dv-processing/io/camera/discovery.hpp>
#include <dv-processing/visualization/event_visualizer.hpp>
#include <dv-processing/data/generate.hpp>
#include <dv-processing/noise/background_activity_noise_filter.hpp>
#include <dv-processing/noise/frequency_filters.hpp>
#include <dv-processing/core/core.hpp>
#include <dv-processing/camera/calibration_set.hpp>

#include <opencv2/highgui.hpp>

#include <queue>
#include <iostream>
#include <condition_variable>
#include <future>

#include <cnpy.h>

using namespace std::chrono_literals;
using namespace std;

int main() {

    // Open the stereo camera with camera names from calibration
    auto leftCamera  = dv::io::camera::open("DXM00089");

    // Make sure both cameras support event stream output, throw an error otherwise
    if (!leftCamera->isEventStreamAvailable()) {
        throw dv::exceptions::RuntimeError("Input camera does not provide an event stream.");
    }

    vector<dv::IMU> imuReadings;

    auto now = chrono::high_resolution_clock::now();

    cout << "Reading from IMU . . ." << endl;
    while (leftCamera->isRunning() && chrono::high_resolution_clock::now() - now < 60s) {
        if (const auto imuBatch = leftCamera->getNextImuBatch())
            imuReadings.insert(imuReadings.end(), imuBatch->begin(), imuBatch->end());
    }
    cout << "Calculating bias . . ." << endl;

    double gyroX, gyroY, gyroZ, accelX, accelY, accelZ = 0;
    for(auto imu : imuReadings) {
        gyroX += imu.gyroscopeX;
        gyroY += imu.gyroscopeY;
        gyroZ += imu.gyroscopeZ;
        accelX += imu.accelerometerX;
        accelY += imu.accelerometerY + 1.0;
        accelZ += imu.accelerometerZ;
    }
    gyroX /= imuReadings.size();
    gyroY /= imuReadings.size();
    gyroZ /= imuReadings.size();
    accelX /= imuReadings.size();
    accelY /= imuReadings.size();
    accelZ /= imuReadings.size();

    cout << "Gyroscope X bias: " << gyroX << endl;
    cout << "Gyroscope Y bias: " << gyroY << endl;
    cout << "Gyroscope Z bias: " << gyroZ << endl;
    cout << "Accelerometer X bias: " << accelX << endl;
    cout << "Accelerometer Y bias: " << accelY << endl;
    cout << "Accelerometer Z bias: " << accelZ << endl;

    return 0;
}