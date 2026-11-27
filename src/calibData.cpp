#include <calibData.h>

namespace Fenrir {
    using namespace std;

    CalibData loadCalibData(string calibJSONPath) {

        auto calibration = dv::camera::CalibrationSet::LoadFromFile(calibJSONPath);

        // Get calibration data
        auto leftCameraCalib = calibration.getCameraCalibration("C0").value();
        auto rightCameraCalib = calibration.getCameraCalibration("C1").value();
        const cv::Size resolution = leftCameraCalib.resolution;

        cv::Mat camMatL_mat = cv::Mat(leftCameraCalib.getCameraMatrix());  // CV_32F
        cv::Mat camMatR_mat = cv::Mat(rightCameraCalib.getCameraMatrix()); // CV_32F

        cv::Mat distCoeffsL_mat = cv::Mat(leftCameraCalib.distortion);     // CV_32F
        cv::Mat distCoeffsR_mat = cv::Mat(rightCameraCalib.distortion);    // CV_32F

        auto Reigen = rightCameraCalib.transformationToC0.getRotationMatrix();
        auto Teigen = rightCameraCalib.transformationToC0.getTranslation();
        cv::Mat R, T;
        cv::eigen2cv(Reigen, R);
        cv::eigen2cv(Teigen, T);

        // Convert everything to CV_64F for stereoRectify
        camMatL_mat.convertTo(camMatL_mat, CV_64F);
        camMatR_mat.convertTo(camMatR_mat, CV_64F);
        distCoeffsL_mat.convertTo(distCoeffsL_mat, CV_64F);
        distCoeffsR_mat.convertTo(distCoeffsR_mat, CV_64F);
        R.convertTo(R, CV_64F);
        T.convertTo(T, CV_64F);

        // Stereo rectify (CV_64F)
        cv::Mat R1, R2, P1, P2, Q;
        stereoRectify(camMatL_mat, distCoeffsL_mat, camMatR_mat, distCoeffsR_mat,
                    resolution, R, T, R1, R2, P1, P2, Q);

        // Convert back to CV_32F for initUndistortRectifyMap
        cv::Mat camMatL_32, camMatR_32, distCoeffsL_32, distCoeffsR_32;
        cv::Mat R1_32, R2_32, P1_32, P2_32;

        camMatL_mat.convertTo(camMatL_32, CV_32F);
        camMatR_mat.convertTo(camMatR_32, CV_32F);
        distCoeffsL_mat.convertTo(distCoeffsL_32, CV_32F);
        distCoeffsR_mat.convertTo(distCoeffsR_32, CV_32F);
        R1.convertTo(R1_32, CV_32F);
        R2.convertTo(R2_32, CV_32F);
        P1.convertTo(P1_32, CV_32F);
        P2.convertTo(P2_32, CV_32F);

        // Generate maps (CV_32F)
        cv::Mat mapL1, mapL2, mapR1, mapR2;
        initUndistortRectifyMap(camMatL_32, distCoeffsL_32, R1_32, P1_32,
                                resolution, CV_32FC1, mapL1, mapL2);
        initUndistortRectifyMap(camMatR_32, distCoeffsR_32, R2_32, P2_32,
                                resolution, CV_32FC1, mapR1, mapR2);

        // Get focal length and baseline from Q
        double focalLength = Q.at<double>(2, 3);
        double baseline = -1.0 / Q.at<double>(3, 2);

        return {
            resolution,
            resolution.height,
            resolution.width,
            leftCameraCalib.name,
            rightCameraCalib.name,
            mapL1,
            mapL2,
            mapR1,
            mapR2,
            focalLength,
            baseline
        };
    }
}