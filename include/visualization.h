#include <opencv2/viz.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>


#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <disparityEstimation.h>

namespace Fenrir {
    using namespace std;

    enum class CvDepth : int {
        U8  = 0,
        I8  = 1,
        U16 = 2,
        I16 = 3,
        I32 = 4,
        F32 = 5,
        F64 = 6,
    };

    pair<CvDepth, int> decodeCvType(int typeCode);

    int cvTypeFromDepth(CvDepth d);

    cv::Mat drawDepthFrame(const vector<StereoBlockMatch>& disparityMeasurements,
                        int searchBound);

    cv::Mat drawDepthColorKey(const cv::Mat& Q, int sbmSearchBound);
}