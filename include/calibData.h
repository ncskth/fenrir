#include <dv-processing/io/camera/discovery.hpp>
#include <dv-processing/core/core.hpp>
#include <dv-processing/camera/calibration_set.hpp>

#include <opencv2/viz.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

namespace Fenrir {
    using namespace std;

    struct CalibData {
        cv::Size resolution;

        // just for less verbose access
        int height;
        int width;

        string leftCameraName;
        string rightCameraName;

        cv::Mat leftUndistortRectifyMap1;
        cv::Mat leftUndistortRectifyMap2;
        cv::Mat rightUndistortRectifyMap1;
        cv::Mat rightUndistortRectifyMap2;

        double focalLength;
        double baseline;
    };

    CalibData loadCalibData(string calibJSONPath);
}