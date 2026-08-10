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

namespace SlamDemo {

using namespace std::chrono_literals;
using namespace std;

const double g = 9.81;
const double degToRad = 3.1415926535897932384626/180.;


tuple<cv::Mat, double> initalRotationFromGravity(vector<dv::IMU> imuReadings, cv::Point3f bias) {
    double dt = 1e-6*((double)(imuReadings.back().timestamp - imuReadings[0].timestamp));

    cv::Vec3d accel(0, 0, 0);

    for (auto imu : imuReadings) {
        accel += cv::Vec3d(imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ);
    }

    // Gram-Schmidt such that gravity becomes our "y" vector
    cv::Vec3d gravity = cv::Vec3d(accel[0]/imuReadings.size(),
                                  accel[1]/imuReadings.size(),
                                  accel[2]/imuReadings.size());
    double gnorm = cv::norm(gravity);
    // reverse direction so y positive is up
    gravity /= -gnorm;
    cv::Vec3d x = cv::Vec3d(1, 0, 0) - gravity[0]*gravity;
    cv::Vec3d z = cv::Vec3d(0, 0, 1) - x[2]*x - gravity[2]*gravity;
    double zn = cv::norm(z);
    z /= zn;

    cv::Mat R = (cv::Mat_<double>(3,3) <<
        x[0], x[1], x[2],
        gravity[0], gravity[1], gravity[2],
        z[0], z[1], z[2]);

    return {R, gnorm};
}


void imuPreintegration(const cv::Point3f gyroBias,
                       const cv::Point3f accelBias,
                       queue<vector<dv::IMU>>& incomingIMU,
                       queue<cv::Mat>& outgoingVelocityVis) {

    while(incomingIMU.empty()) {
        this_thread::sleep_for(1ms);
    }
    vector<dv::IMU> initReadings = incomingIMU.front();
    int64_t lastIMUTime = initReadings[0].timestamp;
    incomingIMU.pop();

    auto gravity = initalRotationFromGravity(initReadings, accelBias);
    cv::Mat initRot = get<0>(gravity);
    double gAcceleration = get<1>(gravity);

    cv::Mat velocityVis = cv::Mat::zeros(400, 400, CV_8UC1);
    cv::Affine3d pose = cv::Affine3d(initRot, cv::Vec3d(0, 0, 0));
    cv::Vec3d worldAccelLowPass(0, 0, 0);
    cv::Vec3d velocity(0, 0, 0);

    while(true) {
        if (!incomingIMU.empty()) {
            //cout << "Received IMU!" << endl;
            vector<dv::IMU> imuBatch = incomingIMU.front();
            incomingIMU.pop();
            //cout << imuBatch.size() << endl;

            cv::Vec3d gyro;
            cv::Vec3d accel;
            cv::Vec3d worldAccel;
            cv::Matx33d R;

            for (auto imu : imuBatch) {
                double dt = 1e-6*(double)(imu.timestamp - lastIMUTime);

                gyro = cv::Vec3d(imu.gyroscopeX - gyroBias.x,
                                 imu.gyroscopeY - gyroBias.y,
                                 imu.gyroscopeZ - gyroBias.z);
                accel = cv::Vec3d(imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ);

                cv::Vec3d deltaAngle = degToRad*gyro*dt;
                R = pose.rotation();
                cv::Matx33d deltaR;
                cv::Rodrigues(deltaAngle, deltaR);
                R = deltaR*R;

                worldAccel = R * accel;  // Rotate to world frame
                worldAccel[1] += gAcceleration;  // Remove gravity
                worldAccelLowPass = 0.99*worldAccelLowPass + 0.01*worldAccel;
                velocity += g*worldAccelLowPass*dt;

                pose = cv::Affine3d(R, pose.translation() + velocity*dt);
                lastIMUTime = imu.timestamp;
            }

            //cout << R << endl;
            //cout << worldAccelLowPass << endl;

            int vel_x_cm = (200 + (int)(velocity[0]))%400;
            vel_x_cm = vel_x_cm < 0 ? 400 + vel_x_cm : vel_x_cm;
            int vel_y_cm = (200 + (int)(velocity[2]))%400;
            vel_y_cm = vel_y_cm < 0 ? 400 + vel_y_cm : vel_y_cm;
            velocityVis.at<uchar>(vel_x_cm, vel_y_cm) = 255;

            //lastIMUTime = imuBatch->back().timestamp;
            outgoingVelocityVis.push(velocityVis);
        } else {
            //cout << "IMU queue size: " << incomingIMU.size() << endl;
            this_thread::sleep_for(2ms);
        }
    }

}
}